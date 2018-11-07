function check_file_extension(file_name, valid_extensions) {
    let ret = false;
    for(let i = 0; i < valid_extensions.length; i++) if (file_name.indexOf(valid_extensions[i]) > -1) {
        ret = true;  // if one extension if in the string
    }
    return ret;
}

function start_video(stream) {
    const video = document.querySelector('#stream video');
    video.src = window.URL.createObjectURL(stream);
    // video.play();
}

function stop_video(stream) {
    const video = document.querySelector('#stream video');
    video.parentNode.removeChild(video);
    video.src = window.URL.revokeObjectURL(stream);
}

function capture(width, height) {
    const canvas = document.querySelector('#grid_canvas');

    const video_el = document.querySelector('#stream video');
    canvas.width = width;
    canvas.height = height;

    // TODO: priority: high -- crop image rather than scale. webcam gets image not square
    let context = canvas.getContext('2d');
    context.drawImage(video_el, 0, 0, width, height);

    let image = context.getImageData(0, 0, width, height);
    let avg = 0;
    for(let i = 0; i < image.data.length; i+=4) {
        avg = (image.data[i] + image.data[i + 1] + image.data[i + 2]) / 3;
        image.data[i] = avg;
        image.data[i + 1] = avg;
        image.data[i + 2] = avg;
        image.data[i + 3] = 255;
    }
    context.putImageData(image, 0, 0);
}

function init_video() {
    // begin video streaming
    // reference: https://www.jonathan-petitcolas.com/2016/08/24/taking-picture-from-webcam-using-canvas.html
    getUserMedia({
        video: true,
        audio: false,
        width: 640,
        height: 480,
        el: 'stream'  // render live video in #stream
    }, stream => {
        start_video(stream);
        document.getElementById('capture').addEventListener('click', () => {
            capture(96, 96);
            // stop_video(stream);
        })
    }, err => console.error(err));
}

let valid_extensions = ['.gif', '.png', '.jpg'];

/*$(document).ready(function() {
    let srcs = '';
    $.ajax({
        async: false,
        url: 'src/srcs.txt',
        dataType: 'text',
        success: function(data) {
            srcs = data;
        }
        });
    srcs = srcs.split('\n');
    let $div = $('#accordion_master');  // will break down to every START::----END::
    let $temp_div = $('<div class="collapse"></div>');  // will break down each file in start::--END::
    for(let i = srcs.length - 1; i >= 0; i --) {
        if(srcs[i].indexOf('START::') > -1) {  // start and stop are switched in meaning when array reversed
            $div.append($temp_div);  // append all individual accordions to big accordion
            $temp_div = $('<div class="collapse"></div>');  // prepare for next ::---:: section
        }else if(srcs[i].indexOf('END::') > -1) {
            $div.append(`<h3>${srcs[i].split('::')[1]}</h3>`);  // add <h3> for head of START::
        }else if(check_file_extension(srcs[i], valid_extensions)) {  // add <h3> and content dev to temp_div
            let file_name = srcs[i].split('/');
            $temp_div.append(`<h3>${file_name[file_name.length - 1]}</h3>`);
            $temp_div.append(`<div><img src="${srcs[i]}"></div>`);
        }
    }
    $('.collapse').accordion({
        heightStyle: 'content',
        collapsible: true
    });

    // video streaming

    $('#init_video_btn').onClick(event, function (event) {
       init_video();
    });
});
*/  // jquery not available when offline
