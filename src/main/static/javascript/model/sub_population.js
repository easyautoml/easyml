function draw_sub_population_confusion(id, data){

    $("#"+id).empty();

    var tbl = document.createElement('table');
    tbl.className = "sub_confusion_matrix";

    // Create Header
    header = "<tr><th></th><th></th> <th colspan="+data.labels.length+"><h4>Actual</h4></th> </tr> <tr> <th></th> <th></th>"
    for(let i=0; i<data.labels.length; i++){
        header = header + "<th><h6>" + data.labels[i] + "</h6></th>"
    }
    header += "</tr>"
    tbl.innerHTML += header

    // Create body
    rows = ""
    for (let i=0; i<data.matrix.length; i++){
        row = document.createElement('tr');

        _row_content = "<tr>"
        for (let j=0; j<data.matrix[i].length; j++){

            if (i==0 & j==0){
                _row_content += "<td rowspan='"+data.matrix[i].length+"'><h4>Predict</h4></td>"
            }

            if (j == 0){
                _row_content += "<td><h6>"+ data.labels[i] +"</h6></td>"
            }

            if (i == j){
                _row_content += "<td style='background: #28a745;'>" + data.matrix[i][j] + "</td>"
            }
            else{
                _row_content += "<td>" + data.matrix[i][j] + "</td>"
            }
        }
        _row_content += "</tr>"

        rows += _row_content
    }

    tbl.innerHTML += rows;

    $("#"+id).append(tbl);

}

function draw_sub_population_density(id, data){

    Highcharts.chart(id, {
        chart: {
            type: 'scatter',
            backgroundColor: '#0c263c',
            width: 500,
            height: 400
        },

        plotOptions: {
            series: {
                lineWidth: 1
            }
        },
        legend: {
            align: 'left',
            verticalAlign: 'top',
            x: 30,
            itemStyle: {
                color:"white",
            },
            y: 20,
            floating: true,

        },
        title: {
            text: ''
        },
        yAxis: {
            title: {
                text: 'Predict Density'
            },
        },
        tooltip: {},
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
                "data": data[0].series,
                zIndex: 1,
                marker: {
                    radius: 0
                },
                name: data[0].name,

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
                "data": data[1].series,
                zIndex: 2,
                marker: {
                    radius: 0
                },
                name: data[1].name,

            }]
    });

}