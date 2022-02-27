$(function() {
  $(document).ready(function() {
    $('#data_source_lst').DataTable();
  });
});

function delete_confirm(delete_file_id, delete_file_name){
    $("#delete_file_id").val(delete_file_id)
    $("#delete_file_name").text(delete_file_name)
}