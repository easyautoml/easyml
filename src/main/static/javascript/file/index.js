function delete_confirm(delete_file_id, delete_file_name){
    $("#delete_file_id").val(delete_file_id)
    $("#delete_file_name").text(delete_file_name)
}

function select_table(table){
    $('#btn_table').text(table);
    $('#selected_table').val(table);
}

$(document).ready(function(){
    $('#upload_file').on('change', function(){
        // Get the file name
        var fileName = $(this).val().replace("C:\\fakepath\\", "");

        // Replace the "Choose a file" label
        $(this).next('.custom-file-label').html(fileName);
    });

    $('#data_source_lst').DataTable();

    // Show selected tab
    if ($('#selected_tab').val() === "connect-db-tab") {
        $('#connect-db-tab').tab('show');
    }
    else if ($('#selected_tab').val() === "upload-tab") {
        $('#upload-tab').tab('show');
    }
    else {
        $('#files-tab').tab('show');
    }

    // Show and hide password
    var passwordInput = $("#pass");
    var passwordIcon = $("#passwordIcon");

    // Toggle password visibility on button click
    $("#showPasswordToggle").click(function () {
        if (passwordInput.attr("type") === "password") {
            passwordInput.attr("type", "text");
            passwordIcon.removeClass("fa-eye");
            passwordIcon.addClass("fa-eye-slash");
        } else {
            passwordInput.attr("type", "password");
            passwordIcon.removeClass("fa-eye-slash");
            passwordIcon.addClass("fa-eye");
        }
    });

    // Hide entities
    $("#data_source_lst_length label").hide();
});
