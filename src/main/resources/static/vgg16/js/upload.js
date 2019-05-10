export function init(onChange, onResponse, onError) {
    document.getElementById( 'file' ).addEventListener( 'change', async (e) => {
        onChange();
        const reader = new FileReader();

        reader.onload = function(e) {
          document.images[0].src = e.target.result;
        }

        reader.readAsDataURL(e.target.files[0]);

        try {
            const res = await uploadImage();
            const json = await res.json();
            onResponse(json);
        } catch(e) {
            console.log(e);
        }

    });

    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
    });
    dropZone.addEventListener('drop', async (ev) => {
      console.log('File(s) dropped');
      onChange();

      // Prevent default behavior (Prevent file from being opened)
      ev.preventDefault();
      const formdata = new FormData();

      if (ev.dataTransfer.items) {
        // Use DataTransferItemList interface to access the file(s)
        for (var i = 0; i < ev.dataTransfer.items.length; i++) {
          // If dropped items aren't files, reject them
          if (ev.dataTransfer.items[i].kind === 'file') {
            var file = ev.dataTransfer.items[i].getAsFile();
            console.log('... file[' + i + '].name = ' + file.name);
            const reader = new FileReader();

            reader.onload = function(e) {
              document.images[0].src = e.target.result;
            }

            reader.readAsDataURL(file);

            formdata.append('file', file);
            break;
          }
        }
      } else {
        // Use DataTransfer interface to access the file(s)
        for (var i = 0; i < ev.dataTransfer.files.length; i++) {
          console.log('... file[' + i + '].name = ' + ev.dataTransfer.files[i].name);
          const reader = new FileReader();

          reader.onload = function(e) {
            document.images[0].src = e.target.result;
          }

          reader.readAsDataURL(ev.dataTransfer.files[i]);

          formdata.append('uploaded_file',  ev.dataTransfer.files[i]);
          break;
        }
      }

      try {
          const res = await uploadImage(formdata);
          if (res.status > 400) {
            onError(res);
          } else {
              const json = await res.json();
              onResponse(json);
          }
      } catch(e) {
          onError(e);
      }

    });
}

function uploadImage(formdata) {
    formdata = formdata || new FormData(document.forms[0]);

    return fetch('/api/image', {
        method: 'post',
        body: formdata
    });
}
