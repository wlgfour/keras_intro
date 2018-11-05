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
    for(let i = 0; i < srcs.length; i ++) {
        if(srcs[i].indexOf('START::') > -1) {
            $('body').append(`<h3>${srcs[i].split('::')[1]}</h3>`)
        }else if(srcs[i].indexOf('END::') > -1) {
            $('body').append('<div class="hr"></div>')
        }else {
            let temp_div = $('<div></div>');
            let file_name = srcs[i].split('/');
            temp_div.append(`<h5>${file_name[file_name.length - 1]}</h5>`);
            temp_div.append(`<img src="${srcs[i]}">`);
            $('body').append(temp_div);
        }
    }
});