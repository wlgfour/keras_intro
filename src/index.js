function check_file_extension(file_name, valid_extensions) {
    let ret = false;
    for(let i = 0; i < valid_extensions.length; i++) if (file_name.indexOf(valid_extensions[i]) > -1) {
        ret = true;  // if one extension if in the string
    }
    return ret;
}

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
    let $div = [];
    for(let i = srcs.length - 1; i >= 0; i --) {
        if(srcs[i].indexOf('START::') > -1) {  // start and stop are switched in meaning when array reversed
            $div.append('<div class="hr"></div>');
            $('body').append($div);
        }else if(srcs[i].indexOf('END::') > -1) {
            $div = $('<div class="section-head"></div>');
            $div.append(`<h3>${srcs[i].split('::')[1]}</h3>`);
        }else if(check_file_extension(srcs[i], valid_extensions)) {
            let temp_div = $('<div></div>');
            let file_name = srcs[i].split('/');
            temp_div.append(`<h5>${file_name[file_name.length - 1]}</h5>`);
            temp_div.append(`<img src="${srcs[i]}">`);
            $div.append(temp_div);
        }
    }
});

// &laquo;
// &raquo;