from re import S
from pyparsing import Each
import streamlit as st

import string

st.title('Aplikasi Enkripsi Dekripsi Super')
st.write("""
# menggunakan vigener chiper dan caesar chiper dan Ganda
""")

#menu
algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    (' - ','GANDA', 'CAESAR', 'VIGENER')
)

menu = st.sidebar.selectbox(
    'Pilih Menu',
    (' - ','ENKRIPSI', 'DEKRIPSI')
)


#enkripsi caesar
abjad = string.printable

def caesar(pesan):
    key1 = key
    chiper = ''
    for i in pesan:
        if i in abjad:
            k = abjad.find(i)
            k = (k + key1)%100
            chiper = chiper+abjad[k]
        else:
            chiper = chiper + 1 
    return chiper

alpabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
alfaindex = dict(zip(alpabet, range(len(alpabet))))
alfaalfa = dict(zip(range(len(alpabet)), alpabet))
def vigenere(pesan, key3):
    chiperV=''
    split = [
        pesan[i : i + len(key3)]
        for i in range(0, len(pesan), len(key3))
    ]
    for ench in split:
        i=0
        for alph in ench:
            number = (alfaindex[alph]+alfaindex[key3[i]]) % len(alpabet)
            chiperV += alfaalfa[number]
            i+=1
    return chiperV
    

#dekripsi
def dcaesar(pesan):
    key1 = key
    chiper = ''
    for i in pesan:
        if i in abjad:
            k = abjad.find(i)
            k = (k - key1)%100
            chiper = chiper+abjad[k]
        else:
            chiper = chiper + 1 
    return chiper

def dvigenere(pesan, key):
    dekripv = ''
    split = [
        pesan[i : i + len(key)]
        for i in range(0, len(pesan), len(key))
    ]
    for ench in split:
        i=0
        for alph in ench:
            number = (alfaindex[alph]-alfaindex[key[i]]) % len(alpabet)
            dekripv += alfaalfa[number]
            i+=1
    return dekripv


if algoritma == ' - ':
    st.write("""
# menggunakan vigener chiper dan caesar chiper dan Ganda
""")
elif algoritma == 'GANDA': 
    if menu == 'ENKRIPSI':
        PlainText = st.text_input('Masukan plaintext')
        key = st.slider('Key', 1, 15)

        st.write(caesar(PlainText))
        a = caesar(PlainText)
        key2 = st.text_input('Key ke-2', ('saya'))
        st.write(vigenere(a, key2))
    else:
        PlainText = st.text_input('Masukan plaintext')
        key2 = st.text_input('Key ke-2', ('saya'))

        st.write(dvigenere(PlainText,key2))
        b = dvigenere(PlainText,key2)
        key = st.slider('Key', 1, 15)
        st.write(dcaesar(b))
elif algoritma == 'CAESAR':
    if menu == 'ENKRIPSI':
        PlainText = st.text_input('Masukan plaintext')
        key = st.slider('Key', 1, 15)

        st.write(caesar(PlainText))
    else :
        PlainText = st.text_input('Masukan plaintext')
        key = st.slider('Key', 1, 15)

        st.write(dcaesar(PlainText))
else:
    if menu == 'ENKRIPSI':
        PlainText = st.text_input('Masukan plaintext')
        key2 = st.text_input('Key ke-2', ('saya'))
        st.write(vigenere(PlainText, key2))
    else:
        PlainText = st.text_input('Masukan plaintext')
        key2 = st.text_input('Key ke-2', ('saya'))
        st.write(dvigenere(PlainText, key2))