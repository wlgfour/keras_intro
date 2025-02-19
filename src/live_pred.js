// WARNING: this is bad code. Read at your own risk


let model;
let points = Array(30);

class Img {
    constructor(image, max) {
        this.img = image;
        this.max = max;
    }
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

    const video_el = document.querySelector('#stream video');
    let v_width = video_el.videoWidth;
    let v_height = video_el.videoHeight;
    let crop_x = 0;
    let crop_y = 0;
    let crop_width = 0;
    if(v_width === v_height) {
        crop_width = crop_x = crop_y = v_height;
    } else if(v_height > v_width) {
        // resize to square with side length width
        crop_y = Math.round((v_height - v_width) / 2);
        crop_x = 0;
        crop_width = v_width;
    } else{  // height smaller than width
        // resize to square side length of height
        crop_x = Math.round((v_width - v_height) / 2);
        crop_y = 0;
        crop_width = v_height;
    }

    canvas.width = 96;
    canvas.height = 96;

    // TODO: priority: high -- crop image rather than scale. webcam gets image not square
    let context = canvas.getContext('2d');
    context.drawImage(video_el, crop_x, crop_y, crop_width, crop_width, 0, 0, 96, 96);

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
    let btn = document.getElementById('init_video_btn');
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
    let context_large = canvas_large.getContext('2d');
    canvas_large.width = 680;
    canvas_large.height = 680;
    context_large.fillStyle = 'rgb(200,0,0)';
    capture();
    const video_el = document.querySelector('#stream video');
    make_prediction_promise().then(function(values) {
        points = values;
        make_prediction();
    });
    let draw_width = 640;
    let draw_height = 480;
    context_large.drawImage(video_el, 0, 0, draw_width, draw_height);
    for(let i = 0; i < points.length; i+=2) {
        // context_large.fillRect(values[i] * 680, values[i + 1] * 680, 5, 5);
        // console.log(values[i], values[i + 1]);
        context_large.fillRect(points[i] * draw_width, points[i + 1] * draw_height, 5, 5);
    }
}

function load_model(path) {
    // let btn = document.getElementById('load_model');
    // btn.parentNode.removeChild(btn);
    tf.loadModel(path).then(function(val) {
        console.log(val);
        $('#make_pred').show();
        model = val;
    });
}

// ----------------------- END video functions ------------------

$(document).ready(function() {
    // -------------- video implementation --------------
    init_video();
    // load_model('keras_face_landmarks/log_dir/v1.1/face_keypoints_v1.1_js/model.json');
    // load versions from txt file
    let srcs = '';
    $.ajax({
        async: false,
        url: '../src/tfjs_versions_facial_keypoints.txt',
        dataType: 'text',
        success: function(data) {
            srcs = data;
        }
        });
    srcs = srcs.split('\n');
    console.log(srcs);
  //   <select>
  //       <option value="volvo">Volvo</option>
  //       ... for every version in txt file
    for(let i = 0; i < srcs.length; i++) {
        $('#model_input').append('<option value="' + srcs[i] + '">' + srcs[i] + '</option>');
    }
});