let model;

class Img {
    constructor(image, max) {
        this.img = image;
        this.max = max;
    }
}

function check_file_extension(file_name, valid_extensions) {
    let ret = false;
    for(let i = 0; i < valid_extensions.length; i++) if (file_name.indexOf(valid_extensions[i]) > -1) {
        ret = true;  // if one extension if in the string
    }
    return ret;
}

//--------------- BEGIN video functions -----------------------

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

function capture() {
    const canvas = document.querySelector('#grid_canvas');
    const canvas_large = document.querySelector('#canvas_large');

    const video_el = document.querySelector('#stream video');
    canvas_large.width = 680;
    canvas_large.height = 680;
    canvas.width = 96;
    canvas.height = 96;

    // TODO: priority: high -- crop image rather than scale. webcam gets image not square
    let context = canvas.getContext('2d');
    let context_large = canvas_large.getContext('2d');
    context_large.drawImage(video_el, 0, 0, 680, 680);
    context.drawImage(video_el, 0, 0, 96, 96);

    let image = context.getImageData(0, 0, 96, 96);
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
    btn = document.getElementById('init_video_btn');
    btn.parentNode.removeChild(btn);
    getUserMedia({
        video: true,
        audio: false,
        width: 640,
        height: 480,
        el: 'stream'  // render live video in #stream
    }, stream => {
        start_video(stream);
        document.getElementById('capture').addEventListener('click', () => {
            capture();
            // stop_video(stream);
        })
    }, err => console.error(err));
}

function preprocess_resize(width, height, image_data) {
    // image_data = context.getImageData().data
    let new_data = [];
    let max = 0;
    // get every 4th element -- inefficient as it is a separate loop
    for(let i = 0; i < 96*96; i++) {
        new_data[i] = image_data[i*4];
        if(image_data[i*4] > max) {
            max = image_data[i*4];
        }
    }
    let data_resize = [];
    let temp = [];
    let pix = 0;
    let ttemp = [];
    for(let i = 0; i < 96; i++) {
        temp.length = 0;
        for(let k = 0; k < 96; k++) {
            ttemp = [];
            ttemp[0] = new_data[pix] / max;
            temp[k] = ttemp;
            pix++;
        }
        data_resize[i] = temp;
    }
    return new Img(data_resize, max);
}

async function make_prediction_promise() {
    let img_el = document.getElementById('grid_canvas');
    let context = img_el.getContext('2d');
    let image = context.getImageData(0, 0, 96, 96);
    let data = image.data;
    let img = preprocess_resize(96, 96, data);
    return await model.predict(tf.tensor([img.img])).data();
}

function make_prediction() {

    const canvas_large = document.querySelector('#canvas_large');
    let context = canvas_large.getContext('2d');
    context.fillStyle = 'rgb(200,0,0)'
    make_prediction_promise().then(function(values) {
        for(let i = 0; i < values.length; i+=2) {
            console.log(values[i], values[i + 1]);
            context.fillRect(values[i] * 680, values[i + 1] * 680, 5, 5);
        }
        capture();
        make_prediction();
    });
}

async function load_model(path) {
    let btn = document.getElementById('load_model');
    btn.parentNode.removeChild(btn);
    model = await tf.loadModel(path);
}

// ----------------------- END video functions ------------------

let valid_extensions = ['.gif', '.png', '.jpg'];

$(document).ready(function() {
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

    // -------------- video functions --------------
    init_video();
    load_model('keras_face_landmarks/log_dir/v1.1/face_keypoints_v1.1_js/model.json');
});

