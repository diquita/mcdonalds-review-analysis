# McDonald’s Store Reviews – NLP-Analyse und Standortvergleich

**IU-Portfolio · Projekt: Data Analysis · Aufgabe 1**

Dieses Projekt untersucht, welche Themen in negativen McDonald’s-Kundenbewertungen auftreten und wie sich deren Verteilung zwischen Filialen unterscheidet. Die Textanalyse erfolgt mit Python. Es werden zwei Verfahren zur Textvektorisierung (**Bag-of-Words** und **TF-IDF**) sowie zwei Verfahren zur Themenextraktion (**LDA** und **NMF**) eingesetzt.

## Datengrundlage

- **Quelle:** [McDonald’s Store Reviews auf Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)
- Der Datensatz enthält Bewertungstexte, Sternebewertungen und Filialadressen.
- Untersucht werden ausschließlich Bewertungen mit **1 oder 2 Sternen**. Diese dienen als Näherung für Kundenbeschwerden, auch wenn nicht jede negative Bewertung eine konkrete Beschwerde enthält.
- Die Original-CSV wird **nicht** mit dem GitHub-Repository veröffentlicht.

Nach dem Herunterladen die Datei unter folgendem Pfad ablegen:

```text
data/McDonald_s_Reviews.csv
```

## Voraussetzungen und Installation (Windows / PowerShell)

Python 3 sowie die Bibliotheken aus `requirements.txt` werden benötigt. Für den Excel-Export wird **XlsxWriter** verwendet; eine separate Installation von `artifact-tool` ist **nicht** erforderlich.

Im Projektverzeichnis ausführen:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ausführung

Standardanalyse mit sechs Themen je Modell:

```powershell
.\.venv\Scripts\python.exe analyse.py --input data/McDonald_s_Reviews.csv --output results
```

Optional kann eine andere Themenanzahl erprobt werden, ohne bisherige Ergebnisse zu überschreiben:

```powershell
.\.venv\Scripts\python.exe analyse.py --input data/McDonald_s_Reviews.csv --output results_8 --topics 8
```

**Achtung:** Bei einer anderen Themenanzahl gelten die für sechs Themen vergebenen Bezeichnungen nicht automatisch. Das Skript kennzeichnet neue Themen daher als zunächst unbenannt.

## Methodisches Vorgehen

1. **Selektion:** Ausschließlich 1- und 2-Sterne-Bewertungen berücksichtigen.
2. **Pre-Processing:** Texte in Kleinbuchstaben umwandeln, URLs und Sonderzeichen entfernen, typische beschädigte Zeichenfolgen bereinigen, identische Kombinationen aus Filialadresse, Sternezahl und bereinigtem Text entfernen und Bewertungen mit weniger als vier Wörtern ausschließen. Englische Stoppwörter werden bei der Vektorisierung gefiltert. Eine Lemmatisierung erfolgt nicht.
3. **Vektorisierung:** Bag-of-Words mittels `CountVectorizer` und TF-IDF mittels `TfidfVectorizer`, jeweils mit Einzelwörtern und Wortpaaren.
4. **Themenextraktion:** LDA auf Bag-of-Words und NMF auf TF-IDF; standardmäßig je sechs Themen (`random_state=42`).
5. **Interpretation:** Die Themen werden anhand der Schlüsselwörter und ausgewählter Beispielbewertungen vorläufig benannt. **LDA- und NMF-Themen sind unabhängig und ihre Nummern nicht gleichzusetzen.**
6. **Standortvergleich:** Für die acht Filialen mit den meisten verwertbaren negativen Bewertungen (mindestens 100) werden die Anteile der jeweils **dominanten NMF-Themen** berechnet und visualisiert.

### Vorläufige Themen des NMF-Modells

| ID | Bezeichnung |
| --- | --- |
| N1 | Drive-through / Wartezeiten |
| N2 | Kundenservice / Personal |
| N3 | Bestellfehler |
| N4 | Langsamer Service |
| N5 | Allgemein negative Erfahrungen |
| N6 | Essensqualität / Temperatur |

Für LDA werden **separate Themenbezeichnungen (L1–L6)** ausgegeben. Das Skript verwendet dieselben NMF-IDs und -Bezeichnungen in Tabellen und Diagrammen, damit die Ergebnisse eindeutig zugeordnet werden können.

## Erzeugte Dateien

Das Skript legt den mit `--output` angegebenen Ordner bei Bedarf an und erzeugt unter anderem:

| Datei | Inhalt |
| --- | --- |
| `analyse_ergebnisse.xlsx` | Formatierte Excel-Datei mit den Tabellenblättern **NMF Themen**, **LDA Themen**, **Standorte** und **Datenqualität** |
| `themenhaeufigkeit.png` | Anzahl negativer Bewertungen nach dominantem NMF-Thema |
| `standortvergleich.png` | Prozentuale Verteilung der NMF-Themen in ausgewählten Filialen |
| `themen_nmf.csv`, `themen_lda.csv` | Separate Themenlisten und Schlüsselwörter |
| `standortvergleich.csv` | Standortvergleich als CSV |
| `beispielbewertungen_nmf.csv` | Drei charakteristische Textbeispiele pro NMF-Thema zur Plausibilitätsprüfung |
| `datenqualitaet.csv` | Kennzahlen der Datenfilterung und -bereinigung |
| `klassifizierte_bewertungen.csv` | Verwertbare Bewertungen mit zugeordneten dominanten Themen |
| `zusammenfassung.json` | Kompakte Metadaten zur Auswertung |

Die CSV-Dateien werden mit **Semikolon als Trennzeichen** exportiert. Für die Betrachtung in deutscher Excel-Umgebung wird die `.xlsx`-Datei empfohlen, da Excel Dezimalzahlen in CSV-Dateien teilweise fälschlich als Datum interpretiert.

## Aufbau des GitHub-Repositories

```text
mcdonalds-review-analysis/
├── analyse.py
├── requirements.txt
├── README.md
└── figures/
    ├── analyse_ergebnisse.xlsx
    ├── themenhaeufigkeit.png
    ├── standortvergleich.png
    ├── beispielbewertungen_nmf.csv
    ├── datenqualitaet.csv
    └── zusammenfassung.json
```

Im Ordner `figures/` liegen **bereits erzeugte Ergebnisse** zur direkten Einsicht. Beim lokalen Start schreibt das Programm neue Dateien in den angegebenen Ausgabeordner, standardmäßig `results/`. Der Datensatz in `data/` und die virtuelle Umgebung `.venv/` bleiben lokal.

## Grenzen und Reflexionspunkte

- Die freiwillig abgegebenen Onlinebewertungen sind **nicht repräsentativ** für alle McDonald’s-Kunden oder Filialbesuche. Aus Themenanteilen lässt sich kein objektives Filialranking ableiten.
- In der Originaldatei kommen **beschädigte Zeichenfolgen** vor. Das Pre-Processing reduziert diese Störungen, kann verlorene Originalzeichen jedoch nicht zuverlässig rekonstruieren.
- Die Modelle können inhaltlich überlappende Themen liefern, etwa Drive-through-Wartezeiten und langsamen Service. Einige Zuordnungen können unpassend sein; die Beispielbewertungen dienen deshalb der kritischen Prüfung.
- Die deutschen Themenbezeichnungen sind **menschliche Interpretationen**, keine vom Modell überprüften Kategorien. Bei Änderungen der Daten oder Parameter müssen sie erneut validiert werden.
- Zur übersichtlichen Darstellung werden für die Häufigkeits- und Standortgrafiken jeweils nur die **dominanten NMF-Themen** gezählt. Nebenanteile anderer Themen gehen in diese Grafiken nicht ein.

## Verwendete Software

Python, pandas, NumPy, scikit-learn, Matplotlib und XlsxWriter. Die installierbaren Abhängigkeiten stehen in `requirements.txt`.
