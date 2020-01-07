# pip install tensorflow==1.14
#nltk.download('punkt')
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle



with open("contenido.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["contenido"]:
        for pattern in intent["patrones"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w!="?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

red_neuronal = tflearn.input_data(shape=[None,len(training[0])])
red_neuronal = tflearn.fully_connected(red_neuronal, 10)
red_neuronal = tflearn.fully_connected(red_neuronal, 10)

red_neuronal = tflearn.fully_connected(red_neuronal,len(output[0]),activation="softmax")
red_neuronal = tflearn.regression(red_neuronal)


modelo= tflearn.DNN(red_neuronal)

try:
    modelo.load("modelo.tflearn")
except:
    modelo.fit(training, output, n_epoch=1000, batch_size=6, show_metric=True)
    modelo.save("modelo.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with Hunab Bot  -> (type exit to stop!)")
    while True:
        inp = input("You: ")
        if inp.lower() == "exit":
            break

        results = modelo.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["contenido"]:
            # print(tg)
            if tg['tag'] == tag:
                print(tg['respuestas'])
                responses = tg['respuestas']
        print(random.choice(responses))



chat()


# print(palabras)
# print(auxX)
# print(auxY)
# print(tags)

# print(entrenamiento)
# print(salida)