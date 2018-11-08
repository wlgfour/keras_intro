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
});

