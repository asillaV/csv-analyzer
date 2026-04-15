# Case Study — Analizzatore CSV

**Autore:** Ingegnere di prodotto con esperienza in sistemi di acquisizione dati e analisi segnali  
**Progetto:** Analizzatore CSV v4  
**Contesto:** Industria — test, collaudo, analisi di sistemi meccanici e sensoristici

---

## 1. Il problema

Nel lavoro quotidiano con sistemi di acquisizione dati, i file prodotti dai sensori — accelerometri, celle di carico, potenziometri — arrivano in formato binario e vengono convertiti in CSV tramite software dedicati.

Il problema non è il formato CSV in sé. Il problema è che **ogni commessa può avere una struttura diversa**: header diversi, nomi colonna diversi, configurazioni diverse, anche quando i sensori di base sono gli stessi. Ogni cliente introduce variazioni. Ogni progetto ha le sue specifiche.

Questo creava due situazioni ricorrenti:

- **Chi sapeva programmare** (Python, LabVIEW) doveva comunque riscrivere o adattare gli script di analisi per ogni nuova commessa. Stesso dominio, stessa logica, codice nuovo ogni volta.
- **Chi non sapeva programmare** usava Excel. Ma Excel non è pensato per dataset acquisiti a 1 kHz o più: limiti sul numero di righe, prestazioni inadeguate su file grandi, impossibilità di fare analisi tecniche complete come FFT o filtraggio.

Il risultato pratico: per ottenere anche solo un plot di un segnale, serviva quasi sempre o scrivere codice o arrangiarsi con uno strumento inadeguato. Attività ripetitive, dipendenza dagli sviluppatori, rallentamenti nella fase di analisi e test.

**Mancava uno strumento abbastanza flessibile da adattarsi a dataset strutturalmente diversi, senza richiedere ogni volta nuova programmazione.**

---

## 2. L'obiettivo

L'obiettivo era costruire uno strumento che un ingegnere tecnico — non necessariamente un programmatore — potesse usare immediatamente su qualsiasi CSV, indipendentemente da come era strutturato.

I requisiti concreti erano:

- Leggere CSV eterogenei senza configurazione manuale della struttura
- Offrire le funzioni di analisi più comuni già integrate (visualizzazione, FFT, filtri, statistiche)
- Essere usabile anche da chi non scrive codice
- Poter essere distribuito senza richiedere installazioni complesse o accesso al codice sorgente
- Essere abbastanza modulare da poter essere esteso o adattato nel tempo

Non era un progetto pensato per sostituire MATLAB o strumenti specialistici. Era pensato per **eliminare la frizione iniziale**: quella fase in cui, prima di fare qualsiasi analisi, bisogna costruire l'ambiente di lavoro da zero.

---

## 3. La soluzione

Analizzatore CSV è una dashboard multi-piattaforma per l'analisi di segnali in file CSV.

Permette di caricare un file CSV — qualunque ne sia la struttura — e passare direttamente all'analisi: visualizzazione dei segnali, filtraggio, FFT, report statistici, export dei risultati. Senza scrivere una riga di codice.

Il software rileva automaticamente la struttura del file (encoding, delimitatore, header, formato numerico) e gestisce i formati più comuni, inclusi quelli con numeri in locale italiana, simboli di valuta, separatori variabili. Una volta caricato il dataset, l'utente sceglie gli assi, seleziona i segnali da analizzare e procede con le operazioni che gli servono.

È disponibile in tre interfacce:
- **Web app** (Streamlit) — la più completa, distribuibile senza installazione
- **Desktop** (Tkinter) — per uso locale classico
- **TUI** (Textual) — per ambienti terminale

---

## 4. Le scelte progettuali

### Architettura modulare: core separato dall'UI

Sin dall'inizio, la logica di elaborazione dati è stata separata dalle interfacce utente. La ragione era pratica: non era chiaro quale forma avrebbe preso il progetto, e mantenere il core indipendente avrebbe permesso di usarlo in contesti diversi — notebook Jupyter, script, app desktop, web app — senza riscrivere la logica di analisi.

Questa scelta ha reso possibile aggiungere la web app senza toccare il motore di elaborazione. Le tre interfacce condividono lo stesso backend.

### Web app come soluzione al problema di distribuzione

Le interfacce TUI e desktop erano nate per uso personale. La svolta verso la web app è arrivata quando si è capito che lo strumento era utile anche ai colleghi, ma distribuire il codice sorgente o gestire installazioni Python su macchine non configurate era impraticabile.

La web app ha risolto entrambi i problemi: nessun codice condiviso, nessuna installazione, soglia di accesso minima.

### Preset: configurazioni riusabili

Durante l'uso quotidiano emergeva un pattern: le stesse impostazioni venivano reinserite manualmente ogni volta. FFT su dati a 1 kHz, filtro passa-basso a 20 Hz di quarto ordine — sempre le stesse combinazioni. Il sistema di preset nasce per eliminare questa ripetizione e per permettere a ogni utente di salvare le configurazioni che usa più spesso. È ancora in fase di consolidamento, ma risponde a un'esigenza reale.

### Quality check non bloccanti

Se il software deve adattarsi a CSV di provenienza eterogenea, deve anche aiutare a capire quando i dati hanno problemi. I quality check (monotonicità dell'asse X, gap nel campionamento, spike anomali) segnalano potenziali criticità senza interrompere il flusso di lavoro. L'utente riceve un avviso, ma può comunque procedere con l'analisi. Questo è utile soprattutto quando il problema non è nell'analisi, ma nel CSV stesso — ad esempio per errori di estrazione dal sistema originario.

---

## 5. Il workflow

Il flusso tipico di una sessione di analisi:

1. **Caricamento** — drag and drop del CSV nell'interfaccia web, o selezione file nel desktop
2. **Lettura automatica** — il software rileva encoding, delimitatore, header, formato numerico
3. **Quality check** — eventuali avvisi su anomalie nei dati (non bloccanti)
4. **Selezione assi** — l'utente sceglie l'asse X (temporale o numerico) e i segnali Y da visualizzare
5. **Visualizzazione** — plot sovrapposti o separati, interattivi
6. **Analisi** — FFT, filtri (Moving Average, Butterworth LP/HP/BP), statistiche per sezione
7. **Export** — plot scaricabili, report HTML interattivi, report visivi in PNG/PDF

Il valore del flusso è nella fase 1→5: dal file grezzo all'analisi visiva in pochi passaggi, senza codice.

---

## 6. Il valore prodotto

### Analisi in tempo reale durante una call con un cliente

Un caso concreto: durante una call tecnica, un cliente sosteneva che il sistema avesse problemi nel calcolo delle accelerazioni, perché la media dei valori non risultava zero. Il cliente aveva fornito un CSV custom — struttura non standard, mai vista prima.

In tempo reale, senza preparazione preventiva:
- il file è stato caricato e letto automaticamente
- è stata eseguita una FFT per verificare il comportamento del segnale
- i dati delle celle di carico sono stati correlati con i dati sull'asse spaziale
- è stato possibile dimostrare che il problema non era nei filtraggi, ma nella posizione errata dello strumento in esercizio

Senza questo strumento, la stessa analisi avrebbe richiesto di interrompere la call, sviluppare uno script ad hoc, riprocessare i dati e riprendere il confronto in un secondo momento.

### Riduzione del tempo di analisi

Attività che in precedenza richiedevano anche alcune ore — capire la struttura del CSV, scrivere o adattare codice, generare i plot — oggi si completano in circa 15 minuti, salvo analisi molto specifiche o personalizzate.

### Abbassamento della barriera tecnica

Il software è usato anche da colleghi non programmatori e da figure junior in onboarding. L'accesso alla web app permette di iniziare a esplorare i dati immediatamente, senza setup, rimandando lo sviluppo di codice solo alle analisi che lo richiedono davvero.

### Analisi in laboratorio più rapide

Durante test su nuovi dispositivi, il software ha permesso di ottenere risultati completi più rapidamente rispetto allo sviluppo di uno script MATLAB iniziale, perché le funzionalità comuni erano già integrate e pronte all'uso.

---

## 7. I limiti attuali

Questi sono i limiti reali che un utente deve conoscere prima di adottare il software:

- **Un solo CSV alla volta**: non è possibile caricare e confrontare più file contemporaneamente nella stessa sessione
- **Nessuna operazione tra dataset**: non si possono combinare, sottrarre o trasformare serie provenienti da file diversi
- **Limite di upload sulla versione cloud**: sulla versione Streamlit Cloud, il limite di upload è circa 200 MB per file
- **Prima lettura lenta su file grandi**: la fase di caricamento e raffinazione può richiedere tempo su dataset pesanti; una volta completata, l'elaborazione è veloce
- **Preset ancora sperimentali**: il sistema funziona, ma ha ancora margini di stabilizzazione

Per analisi che richiedono confronto tra commesse diverse o operazioni complesse tra serie, rimane necessario scrivere codice dedicato.

---

## 8. Evoluzioni possibili

Le priorità di sviluppo emerse dall'uso reale:

1. **Velocità della prima lettura** — la fase di caricamento e raffinazione è il collo di bottiglia principale; c'è margine significativo di miglioramento
2. **Lettura di più CSV contemporaneamente** — permetterebbe di confrontare dataset diversi nello stesso flusso di lavoro, che è uno dei casi d'uso più richiesti
3. **Operazioni tra serie** — combinare, confrontare o trasformare segnali direttamente dentro il tool, senza passare a codice esterno

In parallelo: stabilizzazione del sistema di preset e risoluzione di bug noti nel comportamento del fill/fit su alcuni formati CSV particolari.

---

## Note

Questo case study è scritto sulla base dell'esperienza reale di sviluppo e utilizzo del software. Non contiene metriche inventate né benefici ipotetici. Le situazioni descritte (call con cliente, test in laboratorio, onboarding junior) sono casi d'uso effettivi.

Il progetto è open source e disponibile su GitHub. La web app è distribuibile tramite Streamlit senza installazione lato utente.
