#  Visual RAG : lieux touristiques marocains

Envoyez la **photo d'un monument marocain** : le système le reconnaît, puis rédige une **fiche touristique** (histoire, anecdote, conseils de visite) avec un LLM qui tourne **entièrement en local**.

Projet réalisé lors d'un hackathon *Visual RAG* du module NLP.

## Architecture

```
            Phase hors ligne (indexation)
  dataset/*.png ──► CLIP (ViT-B/32) ──► vecteurs 512D ──► ChromaDB
                                                          (+ métadonnées : nom, ville, histoire)

            Phase en ligne (requête)
  photo ──► CLIP ──► recherche par similarité cosinus dans ChromaDB ──► lieu le plus proche
                                                                           │
                         fiche touristique ◄── LLM local (Ollama) ◄── prompt enrichi (contexte du lieu + question)
```

- **Retrieval** : CLIP projette les images dans un espace latent commun. La photo envoyée est comparée aux images de référence par similarité cosinus.
- **Augmentation** : les métadonnées du lieu trouvé (nom, ville, époque, histoire) sont injectées dans le prompt.
- **Génération** : un LLM local servi par Ollama (Llama 3.2, Qwen 2.5…) rédige la réponse en jouant un guide touristique.

## Lieux couverts

| Lieu | Ville | Images de référence |
|---|---|---|
| Tour Hassan | Rabat | 2 |
| Volubilis | Meknès (région) | 2 |
| Cascades de Sefrou | Sefrou | 2 |
| Aït Benhaddou | Ouarzazate (région) | 1 |
| Medersa Bou Inania | Fès | 1 |

Les descriptions complètes se trouvent dans [`metadata.json`](metadata.json).

## Résultats

Évaluation sur les 4 images de [`test_images/`](test_images/) (Precision@1) :

| Image de test | Attendu | Prédit | Score |
|---|---|---|---|
| test_tour_hassan | Tour Hassan | ✅ Tour Hassan | 0.861 |
| test_volubilis | Volubilis | ✅ Volubilis | 0.791 |
| test_sefrou | Cascades de Sefrou | ❌ Aït Benhaddou | 0.798 |
| test_ait_benhaddou | Aït Benhaddou | ❌ Cascades de Sefrou | 0.771 |

**Precision@1 : 50 % (2/4)**, pour une similarité moyenne de 0,805.
Les deux erreurs viennent de la confusion entre Sefrou et Aït Benhaddou. Deux paysages naturels aux tons ocre et vert, avec une ou deux images de référence seulement, restent difficiles à séparer pour CLIP. Ajouter des images de référence plus variées est la piste d'amélioration la plus directe.

À noter aussi : le LLM peut **halluciner** des détails historiques qui ne figurent pas dans le contexte fourni. Le notebook en montre un exemple.

## Installation

Prérequis : [Ollama](https://ollama.com) installé, avec au moins un modèle.

```bash
ollama pull llama3.2
pip install -r requirements.txt
```

1. **Indexer les images** : exécuter le notebook [`visual_rag_tourisme_tp.ipynb`](visual_rag_tourisme_tp.ipynb) jusqu'à la cellule 4. Cela crée la base `chroma_db/`.
2. **Lancer l'application** :
   ```bash
   streamlit run app.py
   ```

L'application détecte automatiquement le modèle Ollama installé. Le modèle CLIP (environ 400 Mo) est téléchargé au premier lancement.

## Structure

```
├── app.py                        # Application Streamlit
├── visual_rag_tourisme_tp.ipynb  # Notebook : théorie, indexation, retrieval, génération, évaluation
├── metadata.json                 # Fiches des lieux (histoire, époque, style, coordonnées…)
├── dataset/                      # Images de référence indexées (un dossier par lieu)
└── test_images/                  # Images de test pour l'évaluation
```

## Technologies

Python · CLIP (sentence-transformers) · ChromaDB · Ollama · Streamlit · LangChain
