from django.shortcuts import render
import tensorflow.python.keras.saving.model_config as tfs
import predictExternal as pe


def main(request):
    ans = "This is a test"
    return render(request, "main.html", {'ans': ans})


def displayResult(request):
    json_file = open('model.json', 'r')
    l_m_json = json_file.read()
    json_file.close()
    loaded_model = tfs.model_from_json(l_m_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
    input_value = request.GET.get('text_box','This is a default value.')
    print(type(pe.predictExternal(input_value, loaded_model)))
    result = (pe.predictExternal(input_value, loaded_model) - 5) * 2
    return render(request, "result.html", {'ans': result})
