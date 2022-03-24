function draw_predict_vs_actual(){
    Highcharts.chart('predict_vs_actual', {
        chart: {
            type: 'scatter',
            zoomType: 'xy',
            backgroundColor: '#041C32',
            width: 500,
            height: 500
        },
        title: {
            style: {
                color : "white",
                fontSize: "15"
            },
            text: 'Predict vs Actual'
        },
        xAxis: {
            title: {
                enabled: true,
                text: 'Actual'
            },
            crosshair: true,
            startOnTick: true,
            endOnTick: true,
            showLastLabel: true
        },
        yAxis: {
            title: {
                text: 'Predict'
            }
        },
        credits: {
            enabled: false
        },
        plotOptions: {
            scatter: {
                marker: {
                    radius: 5,
                    states: {
                        hover: {
                            enabled: true,
                            lineColor: 'rgb(100,100,100)'
                        }
                    }
                },
                states: {
                    hover: {
                        marker: {
                            enabled: false
                        }
                    }
                },
                tooltip: {
                    headerFormat: '',
                    pointFormat: 'Predict {point.y}, Actual : {point.x}'
                }
            }
        },
        series: [{
            name: null,
            showInLegend: false,
            color: 'rgba(0, 255, 255, .5)',
            data: predict_vs_actual_data
        }]
        });

}


function draw_residual(){
    Highcharts.chart('residual', {
    chart: {
        type: 'scatter',
        zoomType: 'xy',
        backgroundColor: '#041C32',
        width: 500,
        height: 500
    },
    title: {
        style: {
            color : "white",
            fontSize: "15"
        },
        text: 'Residual'
    },
    credits: {
        enabled: false
    },
    xAxis: {
        title: {
            enabled: true,
            text: 'Predict'
        },
        crosshair: true,
        startOnTick: true,
        endOnTick: true,
        showLastLabel: true
    },
    yAxis: {
        title: {
            text: 'Residual'
        }
    },
    plotOptions: {
        scatter: {
            marker: {
                radius: 5,
                states: {
                    hover: {
                        enabled: true,
                        lineColor: 'rgb(100,100,100)'
                    }
                }
            },
            states: {
                hover: {
                    marker: {
                        enabled: false
                    }
                }
            },
            tooltip: {
                headerFormat: '',
                pointFormat: 'Residual {point.y}, Predict : {point.x}'
            }
        }
    },
    series: [{
        name: null,
        showInLegend: false,
        color: 'rgba(0, 255, 255, .5)',
        data: residual
    }]
    });
}