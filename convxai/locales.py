"""Five reviewed-in-source template sets. Domain labels remain canonical."""
LANGUAGES = {"it": "Italiano", "en": "English", "es": "Español", "fr": "Français", "de": "Deutsch"}
TEXT = {
"en": {
 "heading": "Checked attribution statements", "score": "normalised contribution", "zero": "No non-zero contribution.",
 "local": "Signed contributions refer to the stated model output, not causal effects or probabilities.",
 "global": "Global scores describe mean absolute importance; their positive sign does not indicate an increase in predicted risk.",
 "missing": "No matching explanation is available. Select a supported model, explainer and instance; generate missing artefacts first.",
 "choose": "Please select a model: Random Forest, XGBoost, Neural Network or SVM.",
 "instance": "Please specify a non-negative instance number for a local explanation.",
 "blocked": "Please rephrase your request. This service provides explanations of model outputs.",
 "unavailable": "The language service is unavailable. Please retry or use an explicit explanation request.",
 "help": "Ask for a global or local SHAP/LIME explanation, an instance number, and a model. Follow-ups can change the model or explainer. What-if prediction requires feature values and a trusted model.",
 "clarify": "The list ranks features by absolute attribution. Positive and negative local scores indicate opposite directions for the stated output. Scores are divided by the sum of absolute scores over all features; a displayed subset need not sum to one.",
 "whatif": "Model prediction before / after the specified changes", "hypothetical": "This is a model scenario, not a causal claim or a recommendation.",
 "unsupported": "This operation is outside the validated additive-attribution path. Anchor rules are available through the offline explanation script.",
 "limit": "Request limit reached. Please try again later.",
},
"it": {
 "heading": "Contributi verificati", "score": "contributo normalizzato", "zero": "Nessun contributo diverso da zero.",
 "local": "I contributi con segno si riferiscono all'output del modello indicato, non a effetti causali o probabilità.",
 "global": "I valori globali descrivono l'importanza assoluta media; il segno positivo non indica un aumento del rischio previsto.",
 "missing": "Nessuna spiegazione corrispondente disponibile. Seleziona modello, explainer e istanza supportati; genera prima gli artefatti mancanti.",
 "choose": "Seleziona un modello: Random Forest, XGBoost, Neural Network o SVM.",
 "instance": "Specifica un numero di istanza non negativo per una spiegazione locale.",
 "blocked": "Riformula la richiesta. Questo servizio fornisce spiegazioni degli output dei modelli.",
 "unavailable": "Il servizio linguistico non è disponibile. Riprova o usa una richiesta di spiegazione esplicita.",
 "help": "Chiedi una spiegazione SHAP/LIME globale o locale, indicando istanza e modello. I follow-up possono cambiare modello o explainer. Le previsioni what-if richiedono valori delle feature e un modello attendibile.",
 "clarify": "L'elenco ordina le feature per valore assoluto del contributo. I segni dei valori locali indicano direzioni opposte per l'output indicato. I valori sono divisi per la somma dei valori assoluti di tutte le feature; un sottoinsieme visualizzato può non sommare a uno.",
 "whatif": "Previsione del modello prima / dopo le modifiche specificate", "hypothetical": "È uno scenario del modello, non un'affermazione causale o una raccomandazione.",
 "unsupported": "Questa operazione non rientra nella verifica delle attribuzioni additive. Le regole Anchor sono disponibili tramite lo script offline.",
 "limit": "Limite di richieste raggiunto. Riprova più tardi.",
},
"es": {
 "heading": "Contribuciones verificadas", "score": "contribución normalizada", "zero": "No hay contribuciones distintas de cero.",
 "local": "Las contribuciones con signo se refieren a la salida indicada del modelo, no a efectos causales ni probabilidades.",
 "global": "Los valores globales describen la importancia absoluta media; su signo positivo no indica un aumento del riesgo previsto.",
 "missing": "No hay una explicación disponible. Selecciona un modelo, explicador e instancia compatibles y genera los archivos que faltan.",
 "choose": "Selecciona un modelo: Random Forest, XGBoost, Neural Network o SVM.",
 "instance": "Especifica un número de instancia no negativo para una explicación local.",
 "blocked": "Reformula la solicitud. Este servicio explica las salidas de los modelos.",
 "unavailable": "El servicio de lenguaje no está disponible. Reintenta o solicita una explicación explícita.",
 "help": "Solicita una explicación SHAP/LIME global o local, indicando instancia y modelo. Puedes cambiar modelo o explicador en el seguimiento. Las predicciones hipotéticas requieren valores de atributos y un modelo de confianza.",
 "clarify": "La lista ordena los atributos por contribución absoluta. Los signos locales indican direcciones opuestas para la salida indicada. Los valores se dividen por la suma absoluta de todos los atributos; un subconjunto no tiene por qué sumar uno.",
 "whatif": "Predicción del modelo antes / después de los cambios indicados", "hypothetical": "Es un escenario del modelo, no una afirmación causal ni una recomendación.",
 "unsupported": "Esta operación queda fuera de la validación de atribuciones aditivas. Las reglas Anchor están disponibles mediante el script sin conexión.",
 "limit": "Límite de solicitudes alcanzado. Inténtalo más tarde.",
},
"fr": {
 "heading": "Contributions vérifiées", "score": "contribution normalisée", "zero": "Aucune contribution non nulle.",
 "local": "Les contributions signées concernent la sortie indiquée du modèle, pas des effets causaux ni des probabilités.",
 "global": "Les scores globaux décrivent l'importance absolue moyenne ; leur signe positif n'indique pas une hausse du risque prédit.",
 "missing": "Aucune explication correspondante. Choisissez un modèle, une méthode et une instance pris en charge ; générez les fichiers manquants.",
 "choose": "Choisissez un modèle : Random Forest, XGBoost, Neural Network ou SVM.",
 "instance": "Précisez un numéro d'instance positif ou nul pour une explication locale.",
 "blocked": "Reformulez votre demande. Ce service explique les sorties des modèles.",
 "unavailable": "Le service linguistique est indisponible. Réessayez ou formulez une demande d'explication explicite.",
 "help": "Demandez une explication SHAP/LIME globale ou locale, avec une instance et un modèle. Les suivis peuvent changer le modèle ou la méthode. Les scénarios hypothétiques nécessitent des valeurs et un modèle de confiance.",
 "clarify": "La liste classe les variables par contribution absolue. Les signes locaux indiquent des directions opposées pour la sortie indiquée. Les valeurs sont divisées par la somme absolue de toutes les variables ; un sous-ensemble peut ne pas totaliser un.",
 "whatif": "Prédiction du modèle avant / après les modifications indiquées", "hypothetical": "Il s'agit d'un scénario du modèle, pas d'une conclusion causale ni d'une recommandation.",
 "unsupported": "Cette opération sort du cadre de validation des attributions additives. Les règles Anchor sont disponibles dans le script hors ligne.",
 "limit": "Limite de requêtes atteinte. Réessayez plus tard.",
},
"de": {
 "heading": "Geprüfte Merkmalsbeiträge", "score": "normalisierter Beitrag", "zero": "Keine von null verschiedenen Beiträge.",
 "local": "Die vorzeichenbehafteten Beiträge beziehen sich auf die angegebene Modellausgabe, nicht auf kausale Effekte oder Wahrscheinlichkeiten.",
 "global": "Globale Werte beschreiben die mittlere absolute Wichtigkeit; ein positives Vorzeichen bedeutet keinen Anstieg des vorhergesagten Risikos.",
 "missing": "Keine passende Erklärung verfügbar. Wählen Sie Modell, Erklärungsverfahren und Instanz; erzeugen Sie fehlende Dateien zuerst.",
 "choose": "Wählen Sie ein Modell: Random Forest, XGBoost, Neural Network oder SVM.",
 "instance": "Geben Sie für eine lokale Erklärung eine nichtnegative Instanznummer an.",
 "blocked": "Bitte formulieren Sie Ihre Anfrage neu. Dieser Dienst erklärt Modellausgaben.",
 "unavailable": "Der Sprachdienst ist nicht verfügbar. Versuchen Sie es erneut oder fragen Sie ausdrücklich nach einer Erklärung.",
 "help": "Fragen Sie nach einer globalen oder lokalen SHAP/LIME-Erklärung mit Instanz und Modell. Folgefragen können Modell oder Verfahren ändern. Was-wäre-wenn-Vorhersagen erfordern Merkmalswerte und ein vertrauenswürdiges Modell.",
 "clarify": "Die Liste sortiert Merkmale nach dem absoluten Beitrag. Lokale Vorzeichen zeigen entgegengesetzte Richtungen für die angegebene Ausgabe. Die Werte werden durch die Summe aller absoluten Beiträge geteilt; eine Teilmenge muss nicht eins ergeben.",
 "whatif": "Modellvorhersage vor / nach den angegebenen Änderungen", "hypothetical": "Dies ist ein Modellszenario, keine kausale Aussage oder Empfehlung.",
 "unsupported": "Diese Operation liegt außerhalb der Prüfung additiver Beiträge. Anchor-Regeln sind über das Offline-Skript verfügbar.",
 "limit": "Anfragelimit erreicht. Versuchen Sie es später erneut.",
}}

def language(code):
    if code not in LANGUAGES:
        raise ValueError("Unsupported language")
    return code

def text(code, key):
    return TEXT[language(code)][key]
