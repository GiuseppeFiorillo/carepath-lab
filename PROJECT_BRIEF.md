# CarePath Lab — Project Brief

**Versione:** 1.5 — 13 novembre 2026
**Responsabile:** sviluppatore autodidatta (portfolio personale)  
**Stato:** Fasi 0–2 completate; Fase 3 avviata: Explainable AI e fairness

## 1. Perché esiste questo progetto

CarePath Lab è un progetto **open-source, educativo e di ricerca applicata** che permette di studiare, valutare e comunicare il rischio di riammissione ospedaliera entro 30 giorni su un dataset storico pubblico.

**Obiettivo didattico:** costruire e comprendere un ciclo ML completo: comprensione del problema, data quality, SQL, statistica, feature engineering, training, validation, explainability, fairness, API, test, deployment e monitoraggio. Come estensione facoltativa: FHIR e assistente agentico dedicato all'analisi dei risultati.

**Obiettivo portfolio:** dimostrare di saper difendere ogni scelta metodologica, non soltanto esibire codice scritto con AI. Le conoscenze sono trasferibili a ruoli Healthcare AI Data Scientist / Applied ML Engineer.

## 2. Problema, domanda scientifica e limiti

Domanda principale: «Quanto bene riusciamo a stimare il rischio di una riammissione entro 30 giorni e quanto sono affidabili le stime per differenti sottogruppi?»

Domanda operativa di ricerca: «Se fosse possibile analizzare soltanto i primi k casi con rischio stimato più alto, quale frazione delle riammissioni osservate nel test set verrebbe intercettata?»

Il simulatore di capacità confronta, **retrospettivamente**, ranking e selezione casuale con metriche come precision@k e recall@k; **non dimostra** che un intervento reale prevenga ricoveri. Non produce diagnosi, terapie o raccomandazioni cliniche. Non può essere utilizzato per decisioni su pazienti reali: dataset storico, possibili bias, assenza di validazione esterna e prospettica.

## 3. Dataset scelto

**UCI: Diabetes 130-US Hospitals for Years 1999–2008**, ID 296, 101.766 ricoveri, 47 feature indicate nella scheda UCI, 130 ospedali/reti di assistenza. Il dataset contiene dati demografici, diagnostici, amministrativi e relativi ai ricoveri. Licenza **CC BY 4.0** (attribuire la fonte).

- Pagina ufficiale: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Citazione: Clore, Cios, DeShazo, Strack (2014), UCI ML Repository, https://doi.org/10.24432/C5230J
- CSV: `diabetic_data.csv` e `IDS_mapping.csv` (verificare le codifiche sulla documentazione originale).
- Target originario `readmitted`: `<30`, `>30`, `NO`. Target binario **da verificare esplicitamente nel codice**: positivo se `<30`, negativo se `>30` o `NO`.
- Chiavi da trattare come identificatori e **non** come feature predittive: `encounter_id`, `patient_nbr`. Il secondo serve per raggruppare righe dello stesso paziente in train/test, evitando contaminazioni.

**Rischi metodologici da studiare prima del training**:
1. Definire chiaramente il momento della previsione (es. alla dimissione) e controllare che ogni feature esista a quel momento.
2. Analizzare valori assenti, codici sconosciuti, categorie rare, duplicati ed eventuali registrazioni multiple per lo stesso paziente.
3. Esaminare criteri di inclusione/esclusione (ad es. esiti di dimissione incompatibili con una nuova riammissione). Motivare le decisioni, evitare esclusioni opportunistiche e riportarne l'impatto.
4. Usare split **a livello di paziente** (`GroupShuffleSplit` o analogo) prima di adattare imputazioni, encoding o scaling. Usare pipeline sklearn per evitare leakage.
5. Non dichiarare validazione prospettica/temporale o per ospedale se identificativi e date adeguati non sono presenti.
6. Non trattare SHAP come prova di causalità; non interpretare differenze tra gruppi come prova diretta di discriminazione.
7. Le conclusioni sono specifiche del dataset storico e non trasferibili automaticamente alla pratica clinica attuale.

## 4. Funzionalità principali

### MVP A — Comprensione e qualità dei dati
- Script riproducibile di import e validazione (schema, null, categorie, range).
- Caricamento e interrogazioni in PostgreSQL; primi KPI e analisi esplorativa.
- Data dictionary e report delle decisioni di cleaning.

### MVP B — Baseline ML rigorosa
- Regressione logistica con preprocessing in pipeline.
- Split per paziente; validation distinta dal test, test tenuto intatto fino a scelta finale.
- Metriche: prevalence, confusion matrix, precision/recall, AUROC, AP e PR-AUC con convenzione di calcolo esplicita, Brier score, calibrazione e intervalli di confidenza quando appropriati.
- Confronto con Random Forest e LightGBM nella Fase 2, dopo la prima logistica, senza assumere che un modello più complesso vinca.

### MVP C — Explainability e audit
- SHAP/global feature importance con limiti interpretativi.
- Performance e calibrazione per sottogruppi pertinenti, con dimensione dei campioni e incertezza.
- Model Audit Report nella Fase 3: spiegazioni, prestazioni/fairness per sottogruppi, sensibilità alle soglie e limiti. Model card come sintesi successiva di use case, assunzioni e rischi.

### MVP D — Simulatore di capacità
- Ordinamento sul rischio stimato; confronti precision@k, recall@k e baseline random.
- Nessuna stima inventata di ricoveri prevenuti o risparmio sanitario.
- Analisi di sensibilità a soglie/capacità e, se utile, intervalli di confidenza.

### MVP E — Applicazione riproducibile
- API FastAPI e semplice dashboard Streamlit.
- Test con pytest, tracciamento esperimenti MLflow, containerizzazione Docker, CI GitHub Actions.
- Logging e simulazione controllata di data drift; no dati identificativi in log.

### Estensioni facoltative (solo dopo MVP)
- HL7/FHIR con dati **sintetici** generati da Synthea; verificare la compatibilità delle feature invece di simulare predizioni con campi inesistenti.
- Agente per domande *analitiche* con strumenti di sola lettura (SQL/Python) su report aggregati, output strutturati, logging e valutazione. No triage o consulenza medica.

## 5. Stack consigliato e ordine di adozione

**Primi passi:** Python 3.11+, ambiente virtuale, pandas, numpy, matplotlib, scikit-learn, PostgreSQL, Psycopg, unittest e Git. Notebook solo per esplorazione; logica stabile in `carepath_lab/`.

**Confronti completati nella Fase 2:** regressione logistica, Random Forest, LightGBM e CatBoost, con tuning limitato e studio dell'imputazione.

**Per gli approfondimenti successivi:** SHAP e MLflow.

**Dopo primo report robusto:** FastAPI, Pydantic, Streamlit, Docker, GitHub Actions.

**Solo se serve:** cloud, Synthea/FHIR, LangGraph e strumenti LLM. Evitare all'inizio costi API, servizi cloud e complessità infrastrutturale. Il training iniziale funziona su CPU di un normale Mac.

## 6. Criteri di qualità non negoziabili

- Codice comprensibile, modulare, spiegabile e testato.
- Niente metriche presentate senza definizione di split, prevalenza e metodologia.
- Nessuna trasformazione o selezione feature addestrata su dati di validazione/test.
- Risultati riproducibili con random seed, versioni delle librerie e istruzioni chiare.
- Dati, secret, chiavi API e file `.env` fuori dal repository Git.
- README esaustivo con istruzioni, limitazioni e citazione UCI.
- Distinguere **ciò che il modello predice** da **ciò che sarebbe causa o effetto di un intervento**.
- Non pubblicare record di pazienti o dataset con provenienza/licenza incerta.

## 7. Risultato finale desiderato

Un repository GitHub con README, architettura, demo, notebook didattico, pipeline dati, esperimenti riproducibili, API/dashboard, test, audit dei bias e model card. Essere in grado di presentare il progetto in 3 minuti e difendere le decisioni tecniche per altri 20 minuti.

## 8. Fonti principali

- UCI dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Documentazione Codex/ChatGPT: https://learn.chatgpt.com/docs/quickstart
- Codex AGENTS.md: https://learn.chatgpt.com/docs/agent-configuration/agents-md

## 9. Stato di avanzamento

Le Fasi 0–2 sono completate. Ingestione, audit, contratto di coorte/target,
confronti per paziente, imputazione e valutazione conclusiva sulla validation di
sviluppo sono documentati e testati. Il report corrente è `docs/PHASE_2_REPORT.md`;
`docs/BASELINE_REPORT.md` conserva gli esperimenti precedenti. Il test finale non
è stato valutato. README e ROADMAP descrivono comandi e passi successivi;
la Fase 3 riguarda explainability e fairness con un Model Audit Report. Il
tracciamento con MLflow è facoltativo, non un prerequisito per l'audit.
La revisione autorizzata del 20 novembre 2026 introduce un modello corrente
con 41 predittori (`reviewed`), conservando i risultati storici a 45 feature:
motivazioni e confronti sono in `docs/FEATURE_REVIEW.md`.
Gli esercizi iniziali restano in `examples/phase_1/`. Prompt di tutoraggio,
cronologie e riepilogo locale dei progressi sono esclusi da Git.
