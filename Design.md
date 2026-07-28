Design Architetturale e Metodologico

Progetto di Tesi: Analisi del degrado prestazionale in reti Multi-Task (Micro vs Macro Drop)

1. Obiettivo dello Studio

L'esperimento mira a confrontare la resilienza di reti neurali convoluzionali (ResNet, DenseNet, EfficientNet) di fronte a due diverse tipologie di scarsità di dati nel set di addestramento:

Macro Drop (Carenza Quantitativa): Riduzione proporzionale del volume dei dati. Si rimuove una percentuale di campioni da tutte le razze appartenenti a una macroclasse.

Micro Drop (Carenza Qualitativa): Riduzione della varietà dei dati. Si rimuovono intere razze (microclassi) fino a raggiungere lo stesso budget di campioni eliminati nel Macro Drop.

2. Paradigma Architetturale: Topologia Fissa ("Open World")

Per garantire un confronto scientificamente valido tra i due scenari, la rete deve operare in un contesto di "Mondo Aperto" (Open World).

Regola della Topologia: Il numero di nodi di output delle due teste (Micro e Macro) rimane fisso a 37 e 2, indipendentemente dal numero di razze effettivamente presenti nel training set.

Motivazione: In uno scenario reale (Open World), il modello può ricevere in input immagini di cani che non ha mai visto. Se si usasse un'architettura con teste dinamiche (es. riducendo i nodi a 32), il passaggio di un'immagine di una razza ignota causerebbe un crash (Index/Key Error). Con la topologia fissa, il modello tenterà una predizione, sbagliandola (comportamento atteso e misurabile).

3. Gestione degli Split (Train, Validation, Test)

La purezza della valutazione è garantita dal rigoroso isolamento degli insiemi di dati:

Train Set: È l'unico ad essere sottoposto ai processi di Drop (Micro o Macro).

Validation Set: Rimane intatto al 100%. Viene usato dall'early stopping per valutare la generalizzazione del modello sulla vera distribuzione dei dati (Mondo Reale) durante l'addestramento.

Test Set: Rimane intatto al 100%. È la "ground truth" intoccabile per la valutazione finale.

4. Innovazione Valutativa: La Doppia Metrica (Dual Metric)

Poiché nel Micro Drop il modello viene valutato su classi che non ha mai visto, l'F1-Score globale subirà un crollo drastico. Per analizzare a fondo il comportamento della rete, la fase di Test (testmodel.py) estrae due metriche distinte per ogni esecuzione:

Macro F1-Score GLOBALE (Robustezza):

Calcolato su tutte le 37 classi del Test Set.

Risponde alla domanda: "Quanto crolla l'affidabilità del modello nel mondo reale se addestrato senza alcune razze?"

Macro F1-Score CLASSI NOTE (Specializzazione):

Calcolato solo sulle classi sopravvissute al drop nel Training Set.

Il sistema usa il Seed per simulare a vuoto il drop, scoprire quali razze erano state escluse, e filtrare il classification_report di Sklearn calcolando la media F1 solo sulle razze note.

Risponde alla domanda: "Sulle razze che il modello ha effettivamente studiato, mantiene un'alta precisione o il minor volume di dati lo ha danneggiato?"

5. Persistenza e Tracciabilità (Data Logging)

Per agevolare l'analisi dati e la generazione dei grafici per la tesi:

Tutti i log storici delle epoche (Train/Val Loss) vengono centralizzati in un unico file CSV per tipologia di architettura (es. resnet18.csv, densenet.csv).

I file CSV sono scritti in modalità append e contengono le colonne drop_type (micro/macro) e drop_percentage, permettendo un facile filtraggio tramite Pandas per i plot analitici.