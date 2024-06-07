export var basic = (text) => {
    const response = fetch(`http://127.0.0.1:8080/basic?params=${text}`)
    const reader = response.body.getReader();
    let done = false;
    let chunk;

    /*
    return fetch(`http://127.0.0.1:8080/basic?params=${text}`, {
        method: 'GET',
    })
        .then(response =>{
            if(response.status !== 200){
                return false
            }
            console.log(response.body)
            return response
        })
        .catch(error => {
            console.error(error);
            throw error;
        });
    */
}