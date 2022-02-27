const nxtBtn = document.querySelector('#submitBtn');
const form1 = document.querySelector('#form1');
const form2 = document.querySelector('#form2');
const form3 = document.querySelector('#form3');
const form4 = document.querySelector('#form4');
const form5 = document.querySelector('#form5');


const icon1 = document.querySelector('#icon1');
const icon2 = document.querySelector('#icon2');
const icon3 = document.querySelector('#icon3');
const icon4 = document.querySelector('#icon4');
const icon5 = document.querySelector('#icon5');


var viewId = parseInt($("#view_id").val());

$(document).ready(function() {
        displayForms();
        progressBar();
        $("#drp_score_regression").hide();
        $("#drp_score_binary").hide();

    });

function submit(){
    select_split_ratio();
    $("#main_form").submit();
}

function select_file(file_id){
    $('#file_id').val(file_id);
    nextForm();
    $('#main_form').submit();
}

function select_target(column_name, column_id){
    $('#btn_target').text(column_name);
    $('#target_id').val(column_id);
}

function select_problem(problem){
    $('#btn_problem_type').text(problem);
    $('#problem_type').val(problem);
    $('#btn_score').text("Select Score!");

    if(problem == 'regression'){
        $("#drp_score_binary").hide();
        $("#drp_score_regression").show();
    } else {
        $("#drp_score_binary").show();
        $("#drp_score_regression").hide();
    }
}

function select_score(score){
    $('#btn_score').text(score);
    $('#score').val(score);
}

function select_features(){
    var features_id = []
    $('input:checkbox[name=chk_feature]:checked').each(function()
    {
       features_id.push( $(this).val());
    });
    $('#features_id').val(features_id);
}

function select_split_ratio(){
    $('#split_ratio').val($("#btn_split_ratio").val());
}


function nextForm(){
    viewId=viewId+1;
    $("#view_id").val(viewId)
    progressBar();
    displayForms();
}

function prevForm(){
    viewId=viewId-1;
    $("#view_id").val(viewId)
    progressBar();
    displayForms();
}
function progressBar(){
    if(viewId===1){
        icon1.classList.add('active');
        icon2.classList.remove('active');
        icon3.classList.remove('active');
        icon4.classList.remove('active');
        icon5.classList.remove('active');
    }
    if(viewId===2){
        icon1.classList.add('active');
        icon2.classList.add('active');
        icon3.classList.remove('active');
        icon4.classList.remove('active');
        icon5.classList.remove('active');
    }
    if(viewId===3){
        icon1.classList.add('active');
        icon2.classList.add('active');
        icon3.classList.add('active');
        icon4.classList.remove('active');
        icon5.classList.remove('active');
    }
    if(viewId===4){
        icon1.classList.add('active');
        icon2.classList.add('active');
        icon3.classList.add('active');
        icon4.classList.add('active');
        icon5.classList.remove('active');
    }
    if(viewId===5){
        icon1.classList.add('active');
        icon2.classList.add('active');
        icon3.classList.add('active');
        icon4.classList.add('active');
        icon5.classList.add('active');
    }
}

function displayForms(){

    if(viewId>5){
        viewId=1;
        $("#view_id").val(1)
    }

    if(viewId ===1){
        form1.style.display = 'block';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'none';


    }else if(viewId === 2){
        form1.style.display = 'none';
        form2.style.display = 'block';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'none';

    }else if(viewId === 3){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'block';
        form4.style.display = 'none';
        form5.style.display = 'none';
    }else if(viewId === 4){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'block';
        form5.style.display = 'none';

    }else if(viewId === 5){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'block';

    }
}

// for slider

var slider = document.querySelector(".slider");
var output = document.querySelector(".output__value");
output.innerHTML = "Train : " + String(slider.value) + ". Test : " + String(10-slider.value);

slider.oninput = function() {
    output.innerHTML = "Train : " + String(slider.value) + ". Test : " + String(10-slider.value);
}