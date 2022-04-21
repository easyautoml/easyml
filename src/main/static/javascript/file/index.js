function delete_confirm(delete_file_id, delete_file_name){
    $("#delete_file_id").val(delete_file_id)
    $("#delete_file_name").text(delete_file_name)
}

$(document).ready(function(){
    $('#upload_file').on('change',function(){
        //get the file name
        var fileName = $(this).val().replace("C:\\fakepath\\", "");

        //replace the "Choose a file" label
        $(this).next('.custom-file-label').html(fileName);
    });

    $('#data_source_lst').DataTable();

//    for (var i in task_id_list){
//        get_task_result(task_status_list[i], task_id_list[i]);
//    }
});