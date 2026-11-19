"""IU Portfolio: McDonald's-Beschwerden – reproduzierbare NLP-Themenanalyse.

LDA und NMF sind getrennte Modelle: gleicher numerischer Index bedeutet NICHT
inhaltlich das gleiche Thema. Manuelle NMF-Namen sind vorläufig und zu prüfen.
"""
from pathlib import Path
import argparse
import json
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

CUSTOM_STOP = set('mcdonald mcdonalds mc donald donalds restaurant place location store food just really got get went one would could like didnt dont im ive thats wasnt'.split())
STOP = sorted(set(ENGLISH_STOP_WORDS) | CUSTOM_STOP)
# Vorläufige fachliche Zuordnung für die Modellkonfiguration mit genau 6 NMF-Themen.
# Nach jeder Änderung von Datensatz, Reinigung oder Modell müssen sie überprüft werden.
LDA_LABELS_6 = [
    "Bestellungen und Wartezeiten",
    "Personal und technische Probleme",
    "Bestellungen und Produktausgabe",
    "Servicequalität und Umgang",
    "Sauberkeit und Personal",
    "Speisenqualität und Frische",
]

NMF_LABELS_6 = [
    'Drive-through / Wartezeiten',
    'Kundenservice / Personal',
    'Bestellfehler',
    'Langsamer Service',
    'Allgemein negative Erfahrungen',
    'Essensqualität / Temperatur',
]


def clean_text(value):
    text = str(value).lower().replace('\n', ' ')
    # Sichtbare typische Mojibake-/Ersatzzeichenfolgen gezielt entfernen.
    # Einmal verlorene Originalzeichen lassen sich nicht verlässlich herstellen.
    text = re.sub(r'(?:ï¿½)+|(?:\ufffd)+|(?:â€[\x80-\xbf]*)+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def export_excel(path, topics_nmf, topics_lda, sites, metrics, coherence):
    """Export mit dem über PyPI installierbaren Paket XlsxWriter."""
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        book = writer.book
        header = book.add_format({'bold': True, 'font_color': 'white',
                                  'bg_color': '#17365D', 'text_wrap': True,
                                  'valign': 'vcenter', 'border': 0})
        percent = book.add_format({'num_format': '0.0', 'valign': 'top'})
        for name, frame in [('NMF Themen', topics_nmf), ('LDA Themen', topics_lda),
                            ('Standorte', sites), ('Datenqualität', metrics), ('Coherence', coherence)]:
            frame.to_excel(writer, sheet_name=name, index=False, startrow=1, header=False)
            sheet = writer.sheets[name]
            sheet.set_row(0, 34)
            for i, col in enumerate(frame.columns):
                sheet.write(0, i, col, header)
                width = 55 if ('Schlüsselwörter' in col or 'adresse' in col.lower()) else (36 if 'Bezeichnung' in col else 25)
                sheet.set_column(i, i, width, percent if ('Anteil (%)' in col or col.startswith('N') and '–' in col) else None)
            sheet.freeze_panes(1, 0)
            if len(frame):
                sheet.autofilter(0, 0, len(frame), len(frame.columns)-1)


def main():
    ap = argparse.ArgumentParser(description='NLP-Themen aus negativen McDonald’s-Bewertungen')
    ap.add_argument('--input', default='data/McDonald_s_Reviews.csv')
    ap.add_argument('--output', default='results')
    ap.add_argument('--topics', type=int, default=6)
    ap.add_argument('--coherence-words', type=int, default=10, help='Top-Unigramme pro Thema für c_v (Standard: 10)')
    args = ap.parse_args()
    if args.topics < 2:
        ap.error('--topics muss mindestens 2 sein')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, encoding='latin1')
    data.columns = data.columns.str.strip()
    required = {'rating', 'review', 'store_address'}
    if not required.issubset(data.columns):
        raise ValueError(f'CSV-Spalten fehlen: {sorted(required - set(data.columns))}')
    data['stars'] = pd.to_numeric(data['rating'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    negative = data.loc[data.stars.isin([1, 2])].copy()
    # Die Kodierungsprüfung erfolgt am Rohtext vor der Reinigung.
    corrupted = int(negative.review.fillna('').astype(str).str.contains('ï¿½|\ufffd', regex=True).sum())
    negative['clean'] = negative.review.fillna('').map(clean_text)
    n_before = len(negative)
    negative = negative.drop_duplicates(subset=['store_address', 'stars', 'clean']).copy()
    n_dedup = len(negative)
    negative = negative.loc[negative.clean.str.split().str.len() >= 4].copy()
    if negative.empty:
        raise ValueError('Nach der Bereinigung sind keine verwertbaren Texte übrig.')
    print(f'Bewertungen: {len(data)} | negativ: {n_before} | nach Duplikaten: {n_dedup} | verwertbar: {len(negative)}')

    bow = CountVectorizer(stop_words=STOP, min_df=8, max_df=.85, ngram_range=(1, 2), max_features=3500)
    tfidf = TfidfVectorizer(stop_words=STOP, min_df=8, max_df=.85, ngram_range=(1, 2), max_features=3500)
    X_bow = bow.fit_transform(negative.clean)
    X_tfidf = tfidf.fit_transform(negative.clean)
    lda = LatentDirichletAllocation(n_components=args.topics, random_state=42, max_iter=12, learning_method='batch', n_jobs=1)
    nmf = NMF(n_components=args.topics, random_state=42, init='nndsvda', max_iter=350)
    print('Trainiere LDA und NMF ...')
    W_lda = lda.fit_transform(X_bow)
    W_nmf = nmf.fit_transform(X_tfidf)

    def get_terms(model, vectorizer):
        names = vectorizer.get_feature_names_out()
        return [[str(names[i]) for i in weights.argsort()[::-1][:12]] for weights in model.components_]
    words_lda = get_terms(lda, bow)
    words_nmf = get_terms(nmf, tfidf)

    # Vergleichbare Kohärenz: c_v für LDA und NMF auf demselben Textkorpus.
    # Nur Unigramme: das Modell berücksichtigt auch Bigramme, die in tokenisierten
    # Dokumenten ohne Phrasenerkennung nicht als einzelnes Token vorkommen.
    # Nicht vorhandene oder redundante Wörter werden hier nicht künstlich ergänzt.
    texts = [[t for t in txt.split() if t not in STOP] for txt in negative.clean]
    dictionary = Dictionary(texts)
    vocabulary = set(dictionary.token2id)

    def coherence_for(model, vectorizer, method):
        names = vectorizer.get_feature_names_out()
        topics = []
        for weights in model.components_:
            selected = []
            for idx in weights.argsort()[::-1]:
                term = str(names[idx])
                if ' ' not in term and term in vocabulary:
                    selected.append(term)
                if len(selected) >= args.coherence_words:
                    break
            topics.append(selected)
        if any(len(t) < 2 for t in topics):
            raise ValueError('Für die Coherence-Berechnung sind mindestens zwei gültige Begriffe pro Thema erforderlich')
        model_score = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                                     coherence='c_v', processes=1)
        per_topic = model_score.get_coherence_per_topic()
        return topics, [float(v) for v in per_topic], float(model_score.get_coherence())

    print('Berechne Topic Coherence c_v für beide Modelle ...')
    coherence_topics_lda, coherence_lda, coherence_mean_lda = coherence_for(lda, bow, 'LDA')
    coherence_topics_nmf, coherence_nmf, coherence_mean_nmf = coherence_for(nmf, tfidf, 'NMF')
    print(f'Coherence c_v: LDA={coherence_mean_lda:.4f} | NMF={coherence_mean_nmf:.4f}')
    negative['lda_topic'] = W_lda.argmax(axis=1)
    negative['nmf_topic'] = W_nmf.argmax(axis=1)
    # Bezeichnungen sind NICHT algorithmisch ermittelt, sondern redaktionell vergeben.
    labels = NMF_LABELS_6 if args.topics == 6 else ['Nicht benannt (neues Modell)'] * args.topics
    lda_labels = LDA_LABELS_6 if args.topics == 6 else ['Nicht benannt (neues Modell)'] * args.topics
    label_map = {i: f'N{i+1} – {labels[i]}' for i in range(args.topics)}
    negative['nmf_label'] = negative.nmf_topic.map(label_map)
    negative['lda_label'] = negative.lda_topic.map(lambda i: f'L{i+1} – {lda_labels[i]}')

    def topic_table(method, wordlists, assignments):
        count = assignments.value_counts().to_dict()
        return pd.DataFrame([
            {'ID': f'{method[0]}{i+1}', 'Bezeichnung (vorläufig)': (labels if method == 'NMF' else lda_labels)[i],
             'Schlüsselwörter': ', '.join(terms), 'Bewertungen': int(count.get(i, 0)),
             'Anteil (%)': round(100 * count.get(i, 0) / len(negative), 1)}
            for i, terms in enumerate(wordlists)
        ])
    nmf_table = topic_table('NMF', words_nmf, negative.nmf_topic)
    nmf_table['Coherence c_v'] = [round(v, 4) for v in coherence_nmf]
    lda_table = topic_table('LDA', words_lda, negative.lda_topic)
    lda_table['Coherence c_v'] = [round(v, 4) for v in coherence_lda]
    nmf_table.to_csv(output/'themen_nmf.csv', index=False, sep=';', encoding='utf-8-sig')
    lda_table.to_csv(output/'themen_lda.csv', index=False, sep=';', encoding='utf-8-sig')

    counts = negative.groupby('store_address').size().sort_values(ascending=False)
    eligible = counts.loc[counts >= 100].head(8).index
    site = negative.loc[negative.store_address.isin(eligible)]
    share = pd.crosstab(site.store_address, site.nmf_topic, normalize='index').mul(100).reindex(index=eligible, columns=range(args.topics), fill_value=0)
    share.columns = [label_map[i] for i in range(args.topics)]
    share.insert(0, 'Anzahl verwertbarer negativer Bewertungen', counts.loc[eligible].astype(int))
    site_table = share.reset_index().rename(columns={'store_address': 'Filialadresse'})
    for col in list(label_map.values()):
        site_table[col] = site_table[col].round(1)
    site_table.to_csv(output/'standortvergleich.csv', sep=';', index=False, encoding='utf-8-sig')
    negative[['store_address', 'stars', 'review', 'clean', 'lda_label', 'nmf_label']].to_csv(output/'klassifizierte_bewertungen.csv', index=False, sep=';', encoding='utf-8-sig')

    # Beispielbewertungen dienen der qualitativen Validierung der Modellthemen.
    examples = []
    for i in range(args.topics):
        candidates = negative.loc[negative.nmf_topic == i].copy()
        if len(candidates):
            # die drei Bewertungen mit höchstem NMF-Gewicht für dieses Thema
            top_idx = sorted(candidates.index, key=lambda idx: W_nmf[negative.index.get_loc(idx), i], reverse=True)[:3]
            for idx in top_idx:
                examples.append({'Thema': label_map[i], 'Filialadresse': negative.at[idx, 'store_address'],
                                 'Beispielbewertung': str(negative.at[idx, 'review'])[:600]})
    pd.DataFrame(examples).to_csv(output/'beispielbewertungen_nmf.csv', sep=';', index=False, encoding='utf-8-sig')

    metrics = pd.DataFrame([
        ['Bewertungen insgesamt', len(data)], ['Filialadressen insgesamt', data.store_address.nunique()],
        ['Negative Bewertungen (1–2 Sterne)', n_before], ['Negative Texte mit Kodierungsauffälligkeiten', corrupted],
        ['Entfernte Duplikate', n_before-n_dedup], ['Entfernte kurze Bewertungen', n_dedup-len(negative)],
        ['Verwertbare Bewertungen', len(negative)], ['Bag-of-Words-Vokabular', X_bow.shape[1]],
        ['TF-IDF-Vokabular', X_tfidf.shape[1]],
    ], columns=['Kennzahl', 'Wert'])
    coherence_table = pd.DataFrame([
        {'Modell': method, 'Thema': f'{prefix}{i+1}', 'Coherence c_v': round(scores[i], 4),
         'Für Score genutzte Wörter': ', '.join(words[i]), 'Modellmittelwert c_v': round(avg, 4)}
        for method, prefix, scores, words, avg in (
            ('LDA', 'L', coherence_lda, coherence_topics_lda, coherence_mean_lda),
            ('NMF', 'N', coherence_nmf, coherence_topics_nmf, coherence_mean_nmf))
        for i in range(args.topics)
    ])
    coherence_table.to_csv(output/'coherence_scores.csv', sep=';', index=False, encoding='utf-8-sig')
    metrics.to_csv(output/'datenqualitaet.csv', sep=';', index=False, encoding='utf-8-sig')
    summary = {'negative_reviews': n_before, 'usable_reviews': len(negative),
               'encoding_affected_negative_reviews': corrupted, 'topics': args.topics,
               'coherence_metric': 'c_v (Top-Unigramme; identisches tokenisiertes Textkorpus)',
               'coherence_c_v_mean_lda': coherence_mean_lda,
               'coherence_c_v_mean_nmf': coherence_mean_nmf,
               'note': 'NMF- und LDA-Namen sind redaktionell und vorläufig interpretierte Themen. Beide Modelle sind unabhängig; Umbenennungen bei geänderten Daten oder Parametern prüfen.'}
    (output/'zusammenfassung.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Durchgängig identische NMF-IDs und Bezeichnungen in Diagrammen und Tabellen.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh([label_map[i] for i in reversed(range(args.topics))],
            [int((negative.nmf_topic == i).sum()) for i in reversed(range(args.topics))])
    ax.set_xlabel('Anzahl negativer Bewertungen (dominantes NMF-Thema)')
    ax.set_title('Häufigkeit der ermittelten Beschwerdethemen')
    fig.tight_layout()
    fig.savefig(output/'themenhaeufigkeit.png', dpi=180)
    plt.close(fig)

    if len(share):
        plot = share.drop(columns=['Anzahl verwertbarer negativer Bewertungen']).copy()
        # Adresse mit mehreren Bestandteilen besser als nur Städtenamen (die doppelt vorkommen können).
        plot.index = [f'Filiale {i+1}: {address[:42]}' for i, address in enumerate(plot.index)]
        fig, ax = plt.subplots(figsize=(13, 7))
        plot.plot(kind='barh', stacked=True, ax=ax, width=.75)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Anteil der verwertbaren negativen Bewertungen (%)')
        ax.set_ylabel('Filialadresse (gekürzt)')
        ax.set_title('Standortvergleich: dominantes NMF-Thema')
        ax.legend(title='Vorläufige Themenbezeichnung', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
        fig.tight_layout()
        fig.savefig(output/'standortvergleich.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

    export_excel(output/'analyse_ergebnisse.xlsx', nmf_table, lda_table, site_table, metrics, coherence_table)
    print(f'Fertig. Übersicht: {output / "analyse_ergebnisse.xlsx"}')
    print('Hinweis: LDA-/NMF-Bezeichnungen sind interpretative Arbeitstitel, keine automatisch verifizierten Klassifikationen.')

if __name__ == '__main__':
    main()
