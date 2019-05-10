
import { Sketchpad } from './sketchpad.js';

const sketchpad = new Sketchpad({
  element: '#sketchpad',
  width: 200,
  height: 200
});
sketchpad.penSize = 12;

function dataURLtoBlob (dataURL) {
  // Decode the dataURL
  const binary = atob(dataURL.split(',')[1])
  // Create 8-bit unsigned array
  const array = []
  let i = 0
  while (i < binary.length) {
    array.push(binary.charCodeAt(i));
    i++
   }
  // Return our Blob object
  return new Blob([ new Uint8Array(array) ], {type: 'image/png'})
}

//sketchpad.animate(10);

function debounced(delay, fn) {
    let timerId;
    return (...args) => {
        if (timerId) {
            clearTimeout(timerId);
            timerId = null;
        }
        timerId = setTimeout(() => {
            fn(...args);
            timerId = null;
        }, delay);
    };
}

function uploadImage() {
    const canvas = document.getElementById('sketchpad');
    const file = dataURLtoBlob(canvas.toDataURL());
    const formdata = new FormData();
    formdata.append('uploaded_file', file, 'file.png');

    return fetch('/mnist', {
        method: 'post',
        body: formdata
    });
}

let sent = false;
document.getElementById('sketchpad').addEventListener('mouseup', debounced(700, async (e) => {
    if (sent) return;
    sent = true;
    console.log('sending image!');
    const output = document.getElementById('output');
    const loader = document.getElementById('loader');
    loader.classList.add('show');

    try {
        const res = await uploadImage();
        switch (res.status ) {
            case 200:
                const json = await res.json();
                console.log('received response', json.digit);
                output.innerHTML = json.digit;
                break;
            default:
                //
                output.innerHTML = '';
                break;
        }
    } catch(e) {
    }
    loader.classList.remove('show');
    setTimeout(() => {sent = false}, 800);
}));

document.querySelector('[data-action="clear"]').addEventListener('click', (e) => {
    const canvas = document.getElementById('sketchpad');
    const output = document.getElementById('output');

    const context = canvas.getContext('2d');

    context.clearRect(0, 0, canvas.width, canvas.height);

    output.innerHTML = '';
});