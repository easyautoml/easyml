const backgroundColor = "#041C32";
var distribution_chart = null;
var roc_chart = null;
var lift_chart = null;

// distribution_chart.series[0].points[20].highlight()

function threshold_change(){
    var selected_threshold = $('#btn_threshold').val();

    var scores_of_threshold = get_score(selected_threshold);

    roc_chart.series[0].points[selected_threshold].highlight();
    lift_chart.series[0].points[selected_threshold].highlight();
    distribution_chart.series[0].points[selected_threshold].highlight();

    $("#class_score").empty();

    for (key in scores_of_threshold){
        if(["class", "fn", "fp", "tn", "tp", "top_percent_of_predict", "fpr", "ppv", "tpr", "threshold",
        "base_value", "target_population", "overall_population"].includes(key)){
            continue;
        }

        // Create html elements for scores
        var div = document.createElement('div');
        div.className = "item";

        var title = document.createElement('h7');
        title.className = "m-b-20";
        title.textContent = key.replace("_", " ").toUpperCase()

        var val = document.createElement('h4');
        val.className = "text-right"

        if(scores_of_threshold[key] != null){
            val.textContent = scores_of_threshold[key].toFixed(2);
        }else{
            val.textContent = "NaN"
        }
        div.appendChild(title);
        div.appendChild(val);

        $("#class_score").append(div);

        // Add value for confusion matrix
        $("#tp").text(scores_of_threshold["tp"])
        $("#fp").text(scores_of_threshold["fp"])
        $("#fn").text(scores_of_threshold["fn"])
        $("#tn").text(scores_of_threshold["tn"])
    }
}

function select_class(class_id, class_name){
    $('#selected_class').text(class_name);
    $('#selected_class').val(class_id);
    $('#form_selected_class_id').val(class_id);
    $('#form_selected_class_name').val(class_name);
    $('#form_selected_page').val(3);
    $("#form_select_class").submit();

    distribution_chart.series[0].name = "Class : " + class_name;
}

function get_score(selected_threshold){
    for(i in class_scores){

        var score = class_scores[i];

        if (Math.round(score.threshold * 100) == selected_threshold){
            return score
        }
    }
}

function draw_distribution_chart(){
    distribution_chart = Highcharts.chart('distribution_chart', {
        chart: {
            type: 'scatter',
            backgroundColor: '#041C32',
            width: 500,
            height: 500
        },

        plotOptions: {
            series: {
                lineWidth: 1
            }
        },
        legend: {
            align: 'left',
            verticalAlign: 'top',
            x: 100,
            itemStyle: {
                color:"white",
            },
            y: 50,
            floating: true,

        },
        title: {
            style: {
                color : "white",
                fontSize: "15"
            },
            text: 'Prediction Distribution'
        },
        yAxis: {
            title: {
                text: 'Kernel Density'
            },
        },
        tooltip: {
            useHTML: true,
            valueDecimals: 2,
            headerFormat: '<table>',
            pointFormat: '<tr><th>Density :</th><td>{point.y}</td></tr>' +
            '<tr><th>Predict probability:</th><td>{point.x:,.2f}</td></tr>',
            footerFormat: '</table>',
            followPointer: true
        },
        xAxis: {
            title: {
                text: 'Predict Probability'
            },
            reversed: false,

        },
        credits: {
            enabled: false
        },
        series: [
            {
                type: 'area',
                color: '#00FFFF',
                plotOptions: {
                    dataLabels: {
                        enabled: false
                    }
                },
                showInLegend: true,
                "data": predict_distribution.target_class,
                zIndex: 1,
                marker: {
                    radius: 0
                },
                name: 'Target Class',

            },

            {
                type: 'area',
                color: '#696b6c',
                plotOptions: {
                    dataLabels: {
                        enabled: false
                    }
                },
                showInLegend: true,
                "data": predict_distribution.left_class,
                zIndex: 2,
                marker: {
                    radius: 0
                },
                name: 'Left Class',

            }]
    });
}


function draw_roc_chart(){
    roc_chart = Highcharts.chart('roc_chart', {
        chart: {
            type: 'scatter',
            backgroundColor: '#041C32',
            width: 500,
            height: 500
        },

        plotOptions: {
            series: {
                lineWidth: 1
            }
        },
        title: {
            style: {
                color : "white",
                fontSize: "15"
            },
            text: 'ROC Chart'
        },
        tooltip: {
            useHTML: true,
            valueDecimals: 2,
            headerFormat: '<table>',
            pointFormat: '<tr><th>True Positive Rate :</th><td>{point.y}</td></tr>' +
            '<tr><th>False positive rate:</th><td>{point.x:,.2f}</td></tr>',
            footerFormat: '</table>',
            followPointer: true
        },
        yAxis: {
            max: 1,
            title: {
                text: 'True Positive Rate'
            },
        },
        xAxis: {
            max: 1,
            title: {
                text: '1- False Positive Rate'
            },
            reversed: true,
        },
        credits: {
            enabled: false
        },
        series: [
            {
                color: '#00FFFF',
                "data": roc_chart_data,
                zIndex: 2,
                showInLegend: false,
                marker: {
                    radius: 2
                }
            },
            {
                type: 'area',
                color: '#00FFFF',
                plotOptions: {
                    dataLabels: {
                        enabled: false
                    }
                },
                showInLegend: false,
                "data": roc_chart_data,
                zIndex: 1,
                marker: {
                    radius: 0.5
                }

            }]
    });
}

function draw_lift_chart(){
    lift_chart = Highcharts.chart('lift_chart', {
        chart: {
            type: 'scatter',
            backgroundColor: '#041C32',
            width: 500,
            height: 500
        },

        plotOptions: {
            series: {
                lineWidth: 1
            }
        },
        title: {
            style: {
                color : "white",
                fontSize: "15"
            },
            text: 'Lift Chart'
        },
        yAxis: {
            max: 1,
            title: {
                text: 'Target Population'
            },
        },
        tooltip: {
            useHTML: true,
            valueDecimals: 2,
            headerFormat: '<table>',
            pointFormat: '<tr><th>Target population :</th><td>{point.y}</td></tr>' +
            '<tr><th>Overall population:</th><td>{point.x:,.2f}</td></tr>',
            footerFormat: '</table>',
            followPointer: true
        },
        xAxis: {
            max: 1,
            title: {
                text: 'Overall population'
            },
            reversed: false,
        },
        credits: {
            enabled: false
        },
        series: [
            {
                color: '#00ffff',
                "data": lift_chart_data,
                zIndex: 2,
                showInLegend: false,
                marker: {
                    radius: 2
                }
            },
            {
                type: 'area',
                color: '#00FFFF',
                plotOptions: {
                    dataLabels: {
                        enabled: false
                    }
                },
                showInLegend: false,
                "data": lift_chart_data,
                zIndex: 1,
                marker: {
                    radius: 1
                }

            }]
    });
}
