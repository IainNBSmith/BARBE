
function label_symbol_color_set(id, label_val, scope) {
    // buttons follow a basic id - button_#_# - should not regularly be more than 5 digits
    if(/(button_[0-9]+_[0-9]+)/.test(id)){
        if(label_val == null){
            label_val = scope.getElementById(id).innerText
        }
        switch (label_val) {
            // possible values that the button can use as a representation
            case '+': scope.getElementById(id).style.backgroundColor = "green"; break; // slightly positive correlation
            case '++': scope.getElementById(id).style.backgroundColor = "lime"; break; // slightly positive correlation
            case '-': scope.getElementById(id).style.backgroundColor = "red"; break; // slightly negative correlation
            case '--': scope.getElementById(id).style.backgroundColor = "maroon"; break; // slightly positive correlation
            case '.': scope.getElementById(id).style.backgroundColor = "gray"; break; // no meaningful relationship
            default: scope.getElementById(id).style.backgroundColor = "black"; // unknown or unused
        }
    } else if (/button*/.test(id)){
        alert(id)
    }
}

$(document).on('shiny:inputchanged', function(event){
    // check on creation that the color is right
    label_symbol_color_set(event.name, null, this);
});

$(document).on('shiny:updateinput', function(event){
    // check on clicking the element that the color is right
    label_symbol_color_set(event.message.id, event.message.new_label, this);
});