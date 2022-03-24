function draw_pdp_chart(){
    var ymin = null;
    var ymax = null;

    if (problem_type != "regression"){
        ymin = 0;
        ymax = 1;
    }

    Highcharts.chart('pdp_chart', {
        chart: {
            type: 'scatter',
            backgroundColor: '#041C32',
            height: 800
        },
        title: {
            text: ''
        },
        xAxis: [{
            categories: pdp_data.category,
            crosshair: true
        }],
        yAxis: [
        {
            // Primary yAxis
            labels: {
                style: {
                    color: Highcharts.getOptions().colors[0]
                }
            },
            title: {
                text: 'Pdp Values',
                style: {
                    color: Highcharts.getOptions().colors[0]
                }
            },
            max: ymax,
            min: ymin,
        },
        { // Secondary yAxis
            title: {
                text: 'Histogram',
                style: {
                    color: Highcharts.getOptions().colors[1]
                }
            },
            labels: {
                style: {
                    color: Highcharts.getOptions().colors[1]
                }
            },
            opposite: true
        }],
        tooltip: {
            shared: true
        },
        credits: {
            enabled: false
        },
        legend: {
            layout: 'vertical',
            align: 'left',
            x: 100,
            verticalAlign: 'top',
            y: 50,
            floating: true,
            itemStyle: {
                color:"white",
            },
        },
        series: [
        {
            name: 'Histogram',
            type: 'column',
            yAxis: 1,
            data: pdp_data.num,
            color: '#6b6c6c',
        },{
            name: 'Pdp value',
            type: 'spline',
            data: pdp_data.pdp_value,

        }]
    });
}