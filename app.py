import streamlit as st
import pickle, re, numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)

@st.cache_resource
def load_models():
    with open("lr_model.pkl","rb") as f: lr=pickle.load(f)
    with open("tfidf.pkl","rb") as f: tfidf=pickle.load(f)
    classes=np.load("label_classes.npy",allow_pickle=True)
    return lr,tfidf,classes

lr,tfidf,classes=load_models()
stemmer=PorterStemmer()
lemmatizer=WordNetLemmatizer()
STOP=set(stopwords.words("english"))-{"not","no","without"}

def preprocess(text):
    text=str(text).lower()
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"[^a-z0-9\s]"," ",text)
    tokens=[lemmatizer.lemmatize(stemmer.stem(t))
            for t in text.split() if t not in STOP and len(t)>1][:256]
    return " ".join(tokens)

st.set_page_config(page_title="Medical Classifier",page_icon="🏥")
st.title("🏥 Medical Subject Classifier")
st.markdown("Classifies medical questions into 20 subjects.")
st.divider()
examples={
    "Pharmacology":"Patient prescribed metformin type 2 diabetes mechanism Decreases gluconeogenesis Increases insulin secretion",
    "Surgery":"Contraindication laparoscopic cholecystectomy Previous abdominal surgery Pregnancy Acute cholecystitis",
    "Microbiology":"Gram positive cocci clusters coagulase positive beta lactamase Staph aureus Strep pyogenes",
    "Pediatrics":"18 month old fever bulging fontanelle CSF cloudy glucose low Bacterial meningitis Viral meningitis",
    "Anatomy":"Nerve injured fracture surgical neck humerus Axillary Radial Musculocutaneous Ulnar",
}
st.subheader("Quick Examples")
cols=st.columns(len(examples))
selected=""
for col,(label,text) in zip(cols,examples.items()):
    if col.button(label): selected=text
st.divider()
user_input=st.text_area("Enter a medical question:",value=selected,height=150)
if st.button("Classify",type="primary") and user_input.strip():
    clean=preprocess(user_input)
    vec=tfidf.transform([clean])
    proba=lr.predict_proba(vec)[0]
    top5=np.argsort(proba)[::-1][:5]
    st.success("Predicted: "+classes[top5[0]])
    st.metric("Confidence",str(round(proba[top5[0]]*100,1))+"%")
    st.subheader("Top 5 Predictions")
    for i,idx in enumerate(top5):
        st.write(str(i+1)+". "+classes[idx])
        st.progress(float(proba[idx]))
        st.caption(str(round(proba[idx]*100,2))+"%")
st.divider()
st.caption("Group 29 | Variant B | MedMCQA 20-class")