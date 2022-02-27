$(document).ready(function() {
    $( '#predict_form' ).each(function(){
        this.reset();
    });

    new Chart(document.getElementById("chart_models"), {
        type: 'bubble',
        data: {
          labels: "Models Performance",
          datasets: model_performance_data
        },
        options: {
          title: {
            display: false,
            text: ''
          }, scales: {
            yAxes: [{
              scaleLabel: {
                display: true,
                labelString: "Test Score"
              }
            }],
            xAxes: [{
              scaleLabel: {
                display: true,
                labelString: "Evaluation Score"
              }
            }]
          }
        }
    });

    var FImportanceContext = document.getElementById("chart_feature_importance").getContext("2d");
    var FImportanceData = {
        labels: feature_importance_label,
        datasets: [{
            label: "",
            data: feature_importance_data,
            backgroundColor: "#ade8f4",
            hoverBackgroundColor: "#f3722c"
        }]
    };
    var FImportanceChart = new Chart(FImportanceContext, {
        type: 'bar',
        data: FImportanceData,
        options: {
            scales: {
                xAxes: [{
                    ticks: {
                      beginAtZero: true
                    }
                }],
                yAxes: [{
                   stacked: false
                }]
            },
            responsive: true,
            indexAxis: 'y',
            legend: {
                display: false
            },
        }
    });

});

function delete_confirm(delete_predict_id, delete_predict_name){
    $("#delete_predict_id").val(delete_predict_id)
    $("#delete_predict_name").text(delete_predict_name)
}
