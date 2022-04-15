$(document).ready(function() {

    if (evaluation_ready){
        draw_feature_importance();

        if (problem_type == "regression"){
            draw_predict_vs_actual();
            draw_residual();
        } else {
            draw_roc_chart();
            draw_lift_chart();
            draw_distribution_chart();
            threshold_change();

            // Set default for confusion matrix title
            $("#actual_true").text($('#selected_class').text());
            $("#actual_false").text("not " + $('#selected_class').text());
            $("#predict_true").text($('#selected_class').text());
            $("#predict_false").text("not " + $('#selected_class').text());
        }

        if (explain_ready){
            draw_pdp_chart();
        }

    }
    displayForms(selected_page);

});

const form1 = document.querySelector('#form_metric');
const form2 = document.querySelector('#form_feature_importance');
const form3 = document.querySelector('#form_roc');
const form4 = document.querySelector('#form_residual');
const form5 = document.querySelector('#form_sub_population');
const form6 = document.querySelector('#form_pdp');

var slider = document.querySelector(".slider");
var output = document.querySelector(".output__value");
output.innerHTML = "Threshold  : " + String(parseInt(slider.value) / 100);

slider.oninput = function() {
    output.innerHTML = "Threshold  : " + String(slider.value);
    threshold_change();
}

function displayForms(viewId){

    if(viewId ===1){
        form1.style.display = 'block';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'none';
        form6.style.display = 'none';

    }else if(viewId === 2){
        form1.style.display = 'none';
        form2.style.display = 'block';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'none';
        form6.style.display = 'none';
    }else if(viewId === 3){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'block';
        form4.style.display = 'none';
        form5.style.display = 'none';
        form6.style.display = 'none';
    }else if(viewId === 4){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'block';
        form5.style.display = 'none';
        form6.style.display = 'none';
    }else if(viewId === 5){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'block';
        form6.style.display = 'none';

    }else if(viewId === 6){
        form1.style.display = 'none';
        form2.style.display = 'none';
        form3.style.display = 'none';
        form4.style.display = 'none';
        form5.style.display = 'none';
        form6.style.display = 'block';

    }
}

function select_feature(file_metadata_id, column_name){
    $("#selected_column").text(column_name);
    $("#selected_column").val(file_metadata_id);
    $("#form_selected_column_id").val(file_metadata_id);
    $("#form_selected_column_name").val(column_name);
    $("#form_select_feature").submit();
}

function select_pdp_feature(explain_pdp_id, explain_pdp_feature){

    $("#form_selected_explain_pdp_id").val(explain_pdp_id);
    $("#form_selected_explain_pdp_id").text(explain_pdp_id);
    $("#form_selected_explain_pdp_feature").val(explain_pdp_feature);
    $("#form_selected_explain_pdp_feature").text(explain_pdp_feature);

    $("#form_selected_explain_pdp_class").val(null);
    $("#form_selected_explain_pdp_class").text(null);
    $("#form_selected_explain_pdp_class_id").val(null);
    $("#form_selected_explain_pdp_class_id").text(null);

    $("#form_pdp_feature").submit();

}

function select_pdp_class(explain_pdp_class_id, explain_pdp_class_name){
    $("#form_selected_explain_pdp_class").val(explain_pdp_class_name);
    $("#form_selected_explain_pdp_class").text(explain_pdp_class_name);
    $("#form_selected_explain_pdp_class_id").val(explain_pdp_class_id);
    $("#form_selected_explain_pdp_class_id").text(explain_pdp_class_id);
    $("#form_pdp_feature").submit();
}

function draw_feature_importance(){
    Highcharts.chart('feature_importance', {
        chart: {
            type: 'bar',
            zoomType: 'x',
            backgroundColor: '#041C32',
            height: '50%'
        },
        title: {
            text: null
        },
        xAxis: {
            categories: feature_importance.label,
            title: {
                text: null
            },
            labels: {
                style: {
                    color : "white",
                    fontSize: "11"
                }
            },
        },
        yAxis: {
            min: null,
            title: {
                text: null,
                align: 'high'
            },
            labels: {
                overflow: 'justify'
            }
        },
        tooltip: {
            valueSuffix: null
        },
        plotOptions: {
            bar: {
                dataLabels: {
                    enabled: false
                }
            }
        },
        credits: {
            enabled: false
        },
        series: [{
            name: null,
            showInLegend: false,
            color: 'rgba(61, 171, 255, .8)',
            data: feature_importance.data
        }]
    });
}

