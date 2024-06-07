import { useState } from 'react'
import { basic } from './components/Queries'

export default function StaticPage() {

    const [chatHistory, setHistory] = useState([])
    const [chatText, setText] = useState('the isf sldkfj sflkjsd lfsldkjfsdflkj lkjsdf','jhg jhg jhg kjhg jh gljhghg')
    const [chatInput, setInput] =useState('')

    const submitChat = async () => {
      const tempText = chatText
      setText('')
      setHistory([...chatHistory,tempText])
      const response = await fetch(`http://127.0.0.1:8080/basic?params=${String(chatInput)}`)
      setInput('')
      const reader = response.body.getReader();
      const stream = new ReadableStream({
        start(controller) {
          read();

          function read() {
            reader.read().then(({ done, value }) => {
              controller.enqueue(value);
              if (done) {
                controller.close();
              } else {
                read();
              }
            });
          }
        },
      });
      const reader2 = stream.getReader();
      const decoder = new TextDecoder('utf-8');
      while (true) {
        const { done, value } = await reader2.read();
        if (done) {
          break;
        }
        const chunkValue = decoder.decode(value);
        setText((prevData) => prevData + chunkValue)
      }
      
    }

    const handleInput = (event) => {
      setInput(event.target.value)
    }

    const handleEnter = (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submitChat()
      }
    }

  /*  
    const chatList = () => {
      const items = chatHistory.map(item => {
        <li>{item}</li>
      })
      return <ul>{items}</ul>
    }
  */

    return (
        <div id="root">
        <div className='main'>
            <div className="leftBar">
                <div className='leftBarMenu'>
                    <div className='leftTopMenu'>
                        <div className='leftTopIcon'>
                            chats
                        </div>
                        <div className='leftTopIcon'>
                            marks
                        </div>
                    </div>
                    <div className='leftBottomList'>
                        <div className='leftListItem'>
                            <p className='itemText'> This is an item.</p>
                            <div className='itemX'>X</div>
                        </div>
                    </div>
                </div>
            </div>
            <div className='chatScreen'>
                <div className='modelHeader'>
                    <div className='chatTab'>
                        <div>XXX</div>
                        <div>XXX</div>
                        <div>XXX</div>
                    </div>
                    <div className='rounder'>
                        <div className='roundTop'><div className='rTopFill'></div></div>
                        <div className='roundBot'><div className='rBotFill'></div></div>
                    </div>
                    <div className='modelTab'>
                        <div>XXX</div>
                        <div>XXX</div>
                        <div>XXX</div>
                    </div>
                </div>
                <div className='chatBody'>
                    <div className='chatArea'>
                        <div className='listHistory'>{chatHistory.map((item) => (
                          <div className='historyItem'>
                            
                            {item}
                          </div>
                          ))}
                        </div>
                        <div className='newestItem'><div>{chatText}</div></div>
                    </div>
                    <div className='chatEntry'>
                        <textarea
                            type="text"
                            name="userInput"
                            className='entryText'
                            id="text-input"
                            value={chatInput}
                            onChange={handleInput}
                            onSubmit={submitChat}
                            onKeyDown={handleEnter}
                            maxLength={500}
                            minLength={1}
                            required={true}

                            placeholder='Enter chat message here...'
                        />
                        <div onClick={submitChat} className='entrySend'>XXX</div>
                    </div>
                </div>
            </div>
        </div>
        </div>

    )
  }