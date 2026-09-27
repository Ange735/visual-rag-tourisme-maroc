import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import requests
import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_URL = 'http://localhost:11434'

def detecter_modele_ollama(url):
    try:
        r = requests.get(f'{url}/api/tags', timeout=5)
        r.raise_for_status()
        modeles = r.json().get('models', [])
        return modeles[0]['name'] if modeles else None
    except Exception:
        return None

MODELE_DETECTE = detecter_modele_ollama(OLLAMA_URL)

CONFIG = {
    'CHROMA_PATH'     : Path('chroma_db'),
    'COLLECTION_NAME' : 'lieux_touristiques_maroc',
    'CLIP_MODEL'      : 'clip-ViT-B-32',
    'OLLAMA_URL'      : OLLAMA_URL,
    'LLM_MODEL'       : MODELE_DETECTE or 'llama3',
    'N_RESULTS'       : 3,
}

SYSTEM_PROMPT = (
    'Tu es un guide touristique expert specialise dans le patrimoine marocain. '
    'Tu reponds toujours en francais de maniere elegante et pedagogique. '
    'Structure : presentation -> histoire -> anecdote -> conseil visite.'
)

@st.cache_resource
def load_clip():
    return SentenceTransformer(CONFIG['CLIP_MODEL'])

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CONFIG['CHROMA_PATH']))
    return client.get_or_create_collection(
        name=CONFIG['COLLECTION_NAME'],
        metadata={'hnsw:space': 'cosine'},
    )

def rechercher_lieu(image, clip_model, collection, n_results=3):
    embedding = clip_model.encode(image).tolist()
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=['metadatas', 'documents', 'distances'],
    )
    results = []
    for rang, (meta, dist) in enumerate(
        zip(raw['metadatas'][0], raw['distances'][0]), start=1
    ):
        results.append({
            'rang'    : rang,
            'nom'     : meta.get('nom', 'Inconnu'),
            'ville'   : meta.get('ville', 'N/A'),
            'histoire': meta.get('histoire', ''),
            'score'   : round(1 - dist, 4),
        })
    return results

def generate_response(context, question):
    lignes = [
        '=== LIEU IDENTIFIE ===',
        'Nom    : ' + context['nom'],
        'Ville  : ' + context['ville'],
        'Histoire :',
        context['histoire'],
        '====================',
        '',
        'QUESTION : ' + question,
    ]
    prompt = chr(10).join(lignes)
    payload = {
        'model'   : CONFIG['LLM_MODEL'],
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': prompt},
        ],
        'stream'  : False,
        'options' : {'temperature': 0.7, 'num_predict': 512},
    }
    try:
        resp = requests.post(
            CONFIG['OLLAMA_URL'] + '/api/chat', json=payload, timeout=120
        )
        resp.raise_for_status()
        return resp.json()['message']['content']
    except requests.exceptions.ConnectionError:
        return 'Erreur : Ollama inaccessible. Lancez ollama serve.'
    except requests.exceptions.HTTPError as e:
        if '404' in str(e):
            return 'Modele introuvable. Lancez : ollama pull ' + CONFIG['LLM_MODEL']
        return 'Erreur HTTP : ' + str(e)
    except Exception as e:
        return 'Erreur : ' + str(e)

def main():
    st.set_page_config(
        page_title='Visual RAG — Tourisme Marocain',
        page_icon='🕌',
        layout='wide',
    )
    st.title('🕌 Visual RAG — Lieux touristiques marocains')
    st.markdown('Uploadez une photo et obtenez une fiche historique generee par IA.')

    if MODELE_DETECTE is None:
        st.error('Ollama inaccessible. Lancez ollama serve puis ollama pull phi3')
    else:
        st.success('Ollama connecte — modele : ' + MODELE_DETECTE)

    st.divider()

    with st.spinner('Chargement CLIP...'):
        clip_model = load_clip()
    with st.spinner('Connexion ChromaDB...'):
        collection = load_collection()

    n_docs = collection.count()
    if n_docs == 0:
        st.warning('Base vide. Executez la Cellule 4 pour indexer les images.')
        st.stop()
    st.info(f'Base : {n_docs} images indexees')

    col1, col2 = st.columns([1, 1], gap='large')

    with col1:
        st.subheader('Image a identifier')
        uploaded = st.file_uploader('Photo (JPG/PNG)', type=['jpg','jpeg','png'],
                                     label_visibility='collapsed')
        question = st.text_area('Question', value='Parle-moi de ce lieu et donne 3 conseils.', height=80)
        n_results = st.slider('Candidats', 1, 5, 3)
        go = st.button('Identifier', type='primary', use_container_width=True,
                        disabled=(MODELE_DETECTE is None))
        if uploaded:
            st.image(uploaded, use_container_width=True)

    with col2:
        st.subheader('Resultats')
        if go and uploaded:
            image = Image.open(uploaded).convert('RGB')
            with st.spinner('Recherche...'):
                resultats = rechercher_lieu(image, clip_model, collection, n_results)
            if not resultats:
                st.error('Aucun resultat.')
                st.stop()
            for r in resultats:
                st.progress(int(r['score']*100),
                            text=f"#{r['rang']} {r['nom']} ({r['ville']}) — {r['score']:.4f}")
            st.divider()
            lieu = resultats[0]
            st.markdown(f"**Lieu : {lieu['nom']} ({lieu['ville']})**")
            st.caption(f"Confiance : {lieu['score']:.2%}")
            with st.spinner('Generation...'):
                reponse = generate_response(lieu, question)
            st.markdown(reponse)
        elif go:
            st.warning('Uploadez une image.')
        else:
            st.info('Uploadez une image et cliquez sur Identifier.')

    with st.sidebar:
        st.header('Configuration')
        st.json({
            'CLIP'  : CONFIG['CLIP_MODEL'],
            'LLM'   : CONFIG['LLM_MODEL'],
            'Ollama': CONFIG['OLLAMA_URL'],
            'Images': n_docs,
        })
        st.caption('Hackathon Visual RAG 🇲🇦')

if __name__ == '__main__':
    main()