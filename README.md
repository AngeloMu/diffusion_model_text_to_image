# diffusion_model_text_to_image
Questa repository contiene i notebook e i risultati relativi alla parte sperimentale della mia tesi triennale.
Gli esperimenti sono stati condotti con Stable Diffusion v1.5 su Kaggle (GPU T4), usando la libreria Hugging Face Diffusers.

Obiettivi
	•	Analizzare la capacità del modello di generare immagini coerenti rispetto a prompt complessi.
	•	Studiare l’effetto di parametri chiave (seed, guidance scale).
	•	Esplorare la struttura dello spazio latente con traiettorie e interpolazioni.

Esperimenti principali
	1.	Prompt multi-oggetto: verificare se il modello rappresenta correttamente più soggetti nella stessa scena.
	2.	Seed e riproducibilità: osservare come il cambiamento di seed influisce sulla presenza/assenza degli oggetti generati.
	3.	Classifier-Free Guidance (CFG): confronto tra valori bassi, medi e alti per capire il trade-off tra qualità estetica e aderenza al testo.
	4.	Movimento a spirale nel latente: generazioni successive con variazioni minime del seed per evidenziare trasformazioni graduali e coerenza.
	5.	Interpolazione: transizione continua tra due immagini generate a partire da seed diversi della stessa classe (es. uccelli).

Conclusioni
	•	Lo spazio latente mostra continuità e coerenza locale.
	•	Le metriche automatiche non bastano da sole: è necessaria anche valutazione qualitativa.
	•	I parametri come seed e CFG hanno un impatto significativo sul risultato finale.
