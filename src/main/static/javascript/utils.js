function get_task_result(last_status, task_id){

    console.log(task_id)
    const task_url = window.location.origin + "/api/v1/task/?task_id="+task_id;

    fetch(task_url , {
        method: "GET",
        headers: {"X-Requested-With": "XMLHttpRequest",}
    })
    .then(response => response.json())
    .then(data => {
//        console.log("Checked, task status is", data.result.status);
        if (last_status != data.result.status){
            window.location.reload();
        }
        else{
            setTimeout(() => { get_task_result(data.result.status, task_id); }, 4000);
        }
    });
}