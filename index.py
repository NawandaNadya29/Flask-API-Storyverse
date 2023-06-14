from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import pandas as pd

app = Flask(__name__)

# Path ke file dataset
dataset_path = "D:/Bangkit/Capstone Product Based/flaskapi/flask/dataset.csv"

# Variabel global untuk menyimpan dataset, tokenizer, dan model
dataframe = None
tokenizer = None
model = None

# Fungsi untuk memuat dataset saat pertama kali permintaan diterima
def load_dataset():
    global dataframe, tokenizer, model

    # Load dataset
    dataframe = pd.read_csv(dataset_path)
    dataframe = dataframe[['Title', 'Category']]
    dataframe.Title = dataframe.Title.astype(str)  # Mengubah tipe data kolom "Title" menjadi string
    Article = dataframe.Title.values

    # Tokenisasi dataset
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(Article)

    # Load model
    model = load_model('STORYVERSE.h5')

# Endpoint untuk merekomendasikan judul berdasarkan kategori-kategori menggunakan model pelatihan
@app.route('/recommend', methods=['POST'])
def recommend_titles():
    # Memastikan dataset sudah dimuat sebelum melakukan prediksi
    if dataframe is None:
        load_dataset()

    # Mendapatkan kategori-kategori dari body permintaan
    categories = request.json['categories']

    # Mengambil judul-judul dengan kategori-kategori yang diberikan
    category_titles = dataframe[dataframe['Category'].isin(categories)]['Title']

    # Mengubah judul-judul menjadi input yang sesuai dengan tokenizer
    title_sequences = tokenizer.texts_to_sequences(category_titles)
    title_sequences = sequence.pad_sequences(title_sequences, maxlen=110)

    # Melakukan prediksi menggunakan model yang telah dilatih
    predictions = model.predict(title_sequences)

    # Menampilkan judul-judul yang memiliki salah satu atau kedua kategori yang diberikan
    recommended_titles = category_titles.tolist()  # Mengubah DataFrame menjadi list

    # Mengembalikan hasil rekomendasi dalam format JSON
    return jsonify({'recommended_titles': recommended_titles})

if __name__ == '__main__':
    app.run(debug=True)
