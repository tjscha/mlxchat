from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from mlx_lm import load
from utils import generate_step
import time
import mlx.nn as nn
from typing import Union, Optional, Callable, Dict, Generator, Tuple
from transformers import PreTrainedTokenizer
from mlx_lm import tokenizer_utils
#from tokenizer_utils import TokenizerWrapper, top_p_sampling
import time
import mlx.core as mx

app = Flask(__name__)
CORS(app)

model, tokenizer = "",""
#model, tokenizer = load("./models/Meta-Llama-3-70B-4bit")
def load_model(file):
    model, tokenizer = load(file)

@app.route('/test', methods=['GET'])
def tester():
    @stream_with_context
    def stream_gen():
        for i in range(10):
            yield "hi "+str(i)
            time.sleep(.2)
    return Response(stream_gen())

@app.route('/basic', methods=['GET'])
def get_chat():
    user_input = request.args.get('params')
    print(user_input)
    @stream_with_context
    def generate(
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, tokenizer_utils.TokenizerWrapper],
        prompt: str,
        temp: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
        formatter: Optional[Callable] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ):
        """
        Generate text from the model.

        Args:
        model (nn.Module): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (str): The string prompt.
        temp (float): The temperature for sampling (default 0).
        max_tokens (int): The maximum number of tokens (default 100).
        verbose (bool): If ``True``, print tokens and timing information
            (default ``False``).
        formatter (Optional[Callable]): A function which takes a token and a
            probability and displays it.
        repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
        """
        if not isinstance(tokenizer, tokenizer_utils.TokenizerWrapper):
            tokenizer = tokenizer_utils.TokenizerWrapper(tokenizer)

        #if verbose:
        #   print("=" * 10)
        #  print("Prompt:", prompt)

        prompt_tokens = mx.array(tokenizer.encode(prompt))
        detokenizer = tokenizer.detokenizer

        tic = time.perf_counter()
        detokenizer.reset()
        for (token, prob), n in zip(
            generate_step(
                prompt_tokens,
                model,
                temp,
                repetition_penalty,
                repetition_context_size,
                top_p,
                logit_bias,
            ),
            range(max_tokens),
        ):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token)

            if verbose:
                if formatter:
                    # We have to finalize so that the prob corresponds to the last segment
                    detokenizer.finalize()
                    formatter(detokenizer.last_segment, prob.item())
                else:
                    #print(detokenizer.last_segment, end="", flush=True)
                    #print('*', detokenizer.last_segment)
                    yield detokenizer.last_segment

        token_count = n + 1
        detokenizer.finalize()
        if verbose:
            gen_time = time.perf_counter() - tic
            print(detokenizer.last_segment, flush=True)
            print("=" * 10)
            if token_count == 0:
                print("No tokens generated for this prompt")
                return
            prompt_tps = prompt_tokens.size / prompt_time
            gen_tps = (token_count - 1) / gen_time
            print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
            print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        #print("***", detokenizer.text)
        #return detokenizer.text
    return Response(generate(model, tokenizer, prompt=user_input, verbose=True))
    #response.headers['X-Accel-Buffering'] = 'no'
    #return response

@app.route('/', methods=['GET'])
def host():
    return 'mlxchat server running',200

if __name__ == '__main__':
    """
    Start the server
    """
    app.run(host='localhost', port=8080)