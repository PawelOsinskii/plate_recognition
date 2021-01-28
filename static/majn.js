const form = document.getElementById('form');
const steps = document.getElementById('steps');
const result = document.getElementById('text');

form.onsubmit = async (e) => {
    e.preventDefault();
    result.innerHTML = "Loading...";
    fetch('/upload', {
        method: 'POST',
        body: new FormData(form)
    }).then(async response => {
        const {id, text} = await response.json();
        form.remove();
        result.innerHTML = `<h1>Car plate: ${text}</h1>`;
        steps.innerHTML = [1, 2, 3, 4].map(step => {
            return `<h2>Step ${step}</h2><img alt="Step" src="static/images/${id}-step-${step}.jpg"/>`;
        })
    }).catch(e => {
        alert(e)
    });


    return false;
};

