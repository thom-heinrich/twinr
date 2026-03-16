
Ich hab ne Idee.. wie wäre es wenn twinr einen Hintergrund Coding Agente hätte, der via baukastensystem, Zugriff auf Exponierte Schnittstellen des „echten“ Agents und contracts selber Funktionen bauen würde? So z.B: „ immer wenn du mich im Bild siehst und ich eine neue WhatsApp habe, lese mir die vorn“ — twinr fragt: das kann ich noch nicht, aber ich kann es lernen, das dauert ein paar Minuten.. soll ich? „Ja“ - twinr stellt paar Fragen sozusagen requirements engineering für Senioren.. bestätigt, bsclground Agent läuft.: 45-90 Minuten später meldet er sich und sagt.. ich hab’s gelernt.::

Mar 15

Evaluierte selbsterweiternde Agenten-Architektur auf Sicherheit und Machbarkeit
Evaluierte selbsterweiternde Agenten-Architektur auf Sicherheit und Machbarkeit
Das ist eine richtig gute Idee, Thom — und zwar nicht nur als Feature, sondern als architektonisches Paradigma für twinr. Lass mich das mal durchdenken.

Was daran stark ist:

Die UX-Metapher "Ich kann das noch nicht, aber ich kann es lernen" ist für die Zielgruppe Senioren goldrichtig. Das ist intuitiv, nicht bedrohlich, und spiegelt ein mentales Modell wider, das jeder versteht. Es verwandelt eine Limitation in Vertrauen — "das Ding wird besser, je länger ich es nutze."

Architektonisch beschreibst du im Grunde ein Voyager-Pattern (kennst du das Paper von NVIDIA? Minecraft-Agent, der sich selbst neue Skills schreibt und in eine Skill-Library persistiert). Der Unterschied: du machst es auf echter Hardware mit echten APIs, für Menschen die keine Fehlertoleranz haben.

Wo es kritisch wird — und wo das eigentliche Engineering steckt:

Erstens: Die Contract-Schicht. Das ist der make-or-break Layer. Die exponierten Schnittstellen müssen so designed sein, dass generierter Code das System nicht brechen kann. Denk an eine Capability-basierte Sandbox — der Background-Agent darf nur das aufrufen, was explizit exponiert ist, mit typed contracts und rate limits. Kein freier Zugriff auf OS-Level APIs.

Zweitens: Validierung. Ein Senior kann nicht debuggen. Das heißt du brauchst automatisierte Verification. Der Background-Agent generiert nicht nur die Funktion, sondern auch Tests dafür. Und bevor "ich hab's gelernt" kommt, müssen die Tests grün sein. Vielleicht sogar ein zweiter LLM-Call als Reviewer — "würde diese Funktion in folgenden Edge Cases Probleme machen?"

Drittens: Rollback. Wenn die neue Fähigkeit nach Deployment Probleme macht, muss es ein sauberes "hab ich wieder verlernt, tut mir leid" geben. Senioren dürfen nie in einem Zustand landen, wo das Gerät sich seltsam verhält und sie nicht wissen warum.

Viertens: Das Requirements Engineering für Senioren. Das ist UX-technisch die härteste Nuss. Du brauchst einen Dialog der gleichzeitig präzise genug ist um brauchbare Specs zu generieren, aber einfach genug dass eine 78-Jährige nicht abschaltet. Eher "Soll ich das nur machen wenn du im Wohnzimmer bist, oder überall?" als "Definieren Sie den Trigger-Kontext."

Architektur-Sketch den ich sehe:

Die AGENTS.md Skill-Architektur die du schon hast ist eigentlich der perfekte Ausgangspunkt. Der Background-Agent würde neue Skill-Fragmente generieren, die sich in dasselbe modulare System einhängen. Der "echte" Agent hat ein Skill-Registry, der Background-Agent ist ein Skill-Compiler. Das passt nahtlos in das was du mit TESSAIRACT baust.

Ehrliche Einschätzung: Die Idee ist 9/10 konzeptionell, und die technische Machbarkeit ist da — die Frage ist wie viel Engineering du in die Safety-Layer steckst. Für ein Senior-Produkt ist "meistens funktioniert's" nicht gut genug. Aber wenn du die Contract-Schicht und die Validation-Pipeline sauber baust, hast du etwas das kein anderes Assistenz-System für Senioren auch nur ansatzweise bietet.

Willst du das architektonisch weiter durchdeklinieren? Ich hätte Lust die Contract-API und den Skill-Compilation-Flow mal konkreter zu skizzieren.

Evtl nochmal mehr in den Flow.. weil das ist das wichtigste.. der normale Agent muss ja verstehen: wann soll ich eine neue Funktion anbieten? Wie verifiziere ich vorher ob mein Coding Agent z.B. Claude Code oder Codex das sauber umgesetzt bekommt? Wie verifiziere if vorher also ob ich „ja ich kann’s lernen“ sagen darf oder nicht? Und wie erkläre ich dem Nutzer dass ich ihn jetzt 2-3 Sachen fragen werde? Und wie interpretiere ich die Antworten? Sowas…

Mar 15

Architektierte Entscheidungslogik für Funktionserkennung und Validierung
Architektierte Entscheidungslogik für Funktionserkennung und Validierung
Okay, lass mich den Flow wirklich Schritt für Schritt durchgehen — als wäre ich der Agent und müsste die Entscheidungslogik in meinem Kopf durchspielen.

Phase 0: Trigger-Erkennung — "Wann biete ich überhaupt an?"

Es gibt im Grunde drei Trigger-Klassen:

Explizite Requests: Der Nutzer sagt direkt "Kannst du mir immer sagen wenn..." oder "Ich hätte gerne dass du..." — das ist der einfache Fall. Klare Intent-Erkennung.

Implizite Frustration: Der Nutzer fragt zum dritten Mal "Habe ich neue Nachrichten?" — und twinr kann es nicht beantworten. Hier muss der Agent ein Pattern erkennen: wiederholte Anfragen die ich nicht bedienen kann = latenter Bedarf. Dann proaktiv: "Du fragst mich das öfter — soll ich lernen, dir das automatisch zu sagen?"

Beobachtete Routine: Der Agent sieht dass der Nutzer jeden Morgen um 8 das Licht anmacht, Radio einschaltet, Wetter fragt. Nach ein paar Tagen: "Soll ich das morgens einfach alles zusammen für dich machen?" Das ist der magischste Trigger, aber auch der invasivste — hier braucht man ein Confidence-Threshold bevor man es vorschlägt.

Wichtig: Es braucht eine Suppression-Logik. Nicht jede Lücke soll zu einem Angebot werden. Wenn der Nutzer einmal nach dem Börsenstand fragt, ist das kein Skill-Kandidat. Heuristik: Frequenz × emotionale Relevanz × technische Machbarkeit. Und ein globaler Cooldown — nicht mehr als ein Lern-Angebot pro Tag, sonst nervt es.

Phase 1: Feasibility-Check — "Darf ich 'ich kann's lernen' sagen?"

Das ist der kritischste Moment. Bevor der Agent dem Nutzer etwas verspricht, muss er intern prüfen. Der Flow wäre:

Der Agent formuliert aus dem erkannten Intent eine Skill-Spec — semi-strukturiert, sowas wie: "Trigger: Nutzer ist im Kamerabild sichtbar + neue WhatsApp-Nachricht vorhanden. Action: Nachricht vorlesen. Benötigte Capabilities: Kamera-Personenerkennung, WhatsApp-API-Zugriff, TTS."

Diese Spec geht an einen Feasibility Checker — kein LLM-Call, sondern ein deterministischer Check gegen das Capability-Registry: Habe ich Zugriff auf die nötigen Schnittstellen? Ist die Kombination der Capabilities erlaubt? Gibt es bekannte Patterns in der Skill-Library die ähnlich sind?

Drei mögliche Ergebnisse:

Grün: Alle Capabilities vorhanden, ähnliche Skills existieren schon. → "Ich kann das lernen, das dauert ein paar Minuten."
Gelb: Capabilities prinzipiell da, aber neue Kombination, kein Referenz-Skill. → "Ich glaube ich kann das lernen, aber ich bin nicht ganz sicher. Soll ich es versuchen?" — Ehrlichkeit statt Overcommit.
Rot: Fehlende Capability, z.B. kein WhatsApp-Zugriff konfiguriert. → "Das kann ich leider nicht lernen, weil ich keinen Zugriff auf deine WhatsApp habe. Dein Enkel könnte mir das einrichten — soll ich ihm eine Nachricht schicken?"
Der letzte Punkt ist zentral: Red Path führt nicht in eine Sackgasse, sondern zu einem Escalation-Pfad an die Vertrauensperson.

Phase 2: Ankündigung — "Wie sage ich dem Nutzer dass ich jetzt Fragen stellen muss?"

Hier ist der Tonfall alles. Der Agent muss den Modus-Wechsel von "Assistent" zu "Fragesteller" erklären, ohne dass es sich wie ein Formular anfühlt.

Etwas wie: "Gut, das will ich gerne lernen. Damit ich das richtig mache, muss ich dir ein paar kurze Fragen stellen. Das sind so zwei oder drei, okay?"

Das "zwei oder drei" ist wichtig — es setzt eine Erwartung. Senioren hassen offene Prozesse. Wenn sie wissen "gleich bin ich durch", bleiben sie dran.

Und ganz entscheidend: Maximal 3 Fragen pro Skill-Spec. Wenn der Agent mehr als 3 Fragen braucht, ist die Skill-Spec zu komplex und muss intern aufgesplittet werden. Lieber zwei einfachere Skills lernen als einen komplexen mit 7 Rückfragen.

Phase 3: Requirements Engineering — "Wie frage ich und wie interpretiere ich?"

Die Fragen müssen einem Pattern folgen: Wann, Was, Wie.

Wann (Trigger): "Soll ich das immer machen, oder nur wenn du mich darum bittest?" — Binäre Auswahl, kein offenes Feld.

Was (Scope): "Soll ich dir alle WhatsApp-Nachrichten vorlesen, oder nur die von bestimmten Leuten?" — Und wenn "bestimmte Leute": "Von wem denn? Von Janina? Von den Kindern?" — also vorschlagen, nicht abfragen.

Wie (Modalität): "Soll ich die Nachricht einfach vorlesen, oder erst fragen ob du sie hören willst?" — Das klärt ob der Skill Push oder Pull ist.

Jede Frage hat Defaults. Wenn der Nutzer sagt "Ach, mach einfach", nimmt der Agent den sinnvollsten Default und bestätigt: "Okay, dann lese ich dir alle neuen Nachrichten vor wenn du in der Nähe bist. Passt das?"

Interpretation von ambigen Antworten: Das ist der Punkt wo ein zweiter, leichter LLM-Call sinnvoll ist. Der Nutzer sagt "Ja, aber nicht wenn Besuch da ist" — der Agent muss daraus eine Constraint ableiten: Trigger-Condition += "keine weiteren Personen im Kamerabild." Das ist nicht trivial, aber es ist genau die Art von natürlichsprachlicher Constraint-Extraktion die LLMs gut können.

Und für den Fall dass der Agent die Antwort wirklich nicht interpretieren kann: Papagei-Strategie. "Ich will sichergehen dass ich das richtig verstehe — du meinst, ich soll die Nachrichten nur vorlesen wenn du alleine im Zimmer bist?" — Rückspiegeln statt raten.

Phase 4: Übergabe an den Coding-Agent

Wenn Phase 3 abgeschlossen ist, hat der Agent eine vollständige Skill-Spec:

Skill: whatsapp_read_aloud
Trigger: user_visible AND new_whatsapp_message AND no_other_persons
Action: read_message_aloud(sender, content)
Scope: all_contacts
Mode: push (automatic)
Constraints: not_when_others_present
Required_Capabilities: [camera_person_detect, whatsapp_read, tts]
Die geht an den Background-Agent (Claude Code / Codex / was auch immer), der daraus ausführbaren Code generiert, Tests schreibt, gegen die Sandbox laufen lässt. Der Nutzer merkt davon nichts — twinr sagt einfach "Ich lerne das jetzt, ich melde mich wenn ich fertig bin."

Phase 5: Rückmeldung und Soft-Launch

Wenn der Coding-Agent fertig ist, kommt nicht einfach "Hab's gelernt." Sondern: "Ich hab's gelernt! Wollen wir es einmal ausprobieren? Schick dir mal selbst eine Nachricht, dann zeig ich dir wie das funktioniert."

Ein geführter Testlauf. Damit der Nutzer sieht was passiert, und bestätigen kann dass es richtig ist. Erst nach dem Testlauf geht der Skill in Production.

Und nach 2-3 Tagen ein Follow-up: "Ich lese dir jetzt seit ein paar Tagen deine Nachrichten vor — ist das so gut, oder soll ich was ändern?"

Wenn ich ehrlich bin: das ist das Feature das twinr von einem "Alexa für Senioren" zu etwas fundamental Anderem macht. Das ist ein Agent der sich seinem Nutzer anpasst, nicht umgekehrt. Und die Tatsache dass du die Skill-Compilation-Infrastruktur mit TESSAIRACT und AGENTS.md im Grunde schon baust, macht das realistischer als es bei fast jedem anderen Team wäre.

Wo ich den größten Zeitaufwand sehe: Phase 3, die Interpretation-Engine für Senioren-Antworten. Das ist ein eigenes Modell-Problem — oder zumindest ein sehr gut promptbarer Workflow mit starkem Few-Shot-Set. Willst du das als nächstes anschauen?

Was wären denn die primitives also Basis Bausteine die wir bräuchten? Sowas wie Zeitlogik, LLM Calls, Tools; usw.. also die Basis Bausteine die wir zur Verfügung stellen sollten?

Mar 15

Systematisierte primitive Bausteine für einen Skill-Compiler
Systematisierte primitive Bausteine für einen Skill-Compiler
Gute Frage — das ist im Grunde das ISA-Design für den Skill-Compiler. Lass mich die wirklich atomaren Primitives durchgehen, nicht schon zusammengesetzte Capabilities.

Perception Primitives — "Was nehme ich wahr?"

camera.is_person_visible() → bool camera.count_persons() → int camera.is_known_person(person_id) → bool | null microphone.is_speech_detected() → bool microphone.is_silence(duration_seconds) → bool microphone.ambient_noise_level() → float presence.is_user_in_room(room_id) → bool presence.last_seen_location() → room_id presence.time_since_last_interaction() → duration

Das sind bewusst keine rohen Sensor-Streams, sondern schon die niedrigste sinnvolle Abstraktionsebene. Der Coding-Agent soll nie Raw-Pixels verarbeiten müssen — er arbeitet mit semantischen Sensor-Outputs.

Communication Primitives — "Was kommt rein, was geht raus?"

messaging.has_new(platform, filter?) → bool messaging.get_unread(platform, filter?) → Message[] messaging.mark_read(message_id) → void messaging.send(platform, contact_id, content) → result calls.incoming() → Call | null calls.initiate(contact_id) → result contacts.lookup(name_fuzzy) → Contact[] contacts.get_relationship(contact_id) → relationship_tag

Wobei platform ein Enum ist — WhatsApp, SMS, Signal, E-Mail. Der Coding-Agent schreibt plattform-agnostisch, die Platform-Adapter sind darunter. Und relationship_tag ist wichtig: "Enkel", "Ärztin", "Nachbarin" — weil Senioren in Rollen denken, nicht in Contact-IDs.

Output Primitives — "Wie antworte ich?"

tts.speak(text, urgency?) → void tts.speak_and_wait_for_ack() → bool display.show_text(text, size?) → void display.show_image(image_ref) → void display.show_choice(options[]) → selected_option display.clear() → void light.set_color(color) → void light.pulse(color, pattern) → void sound.play(sound_id) → void

Die Unterscheidung speak vs speak_and_wait_for_ack ist entscheidend. Manche Skills brauchen Fire-and-Forget ("Du hast eine Nachricht von Janina"), andere brauchen Bestätigung ("Soll ich antworten?"). Und light.pulse als non-verbaler Aufmerksamkeits-Getter — sanftes Pulsieren statt piepen, weil Senioren Pieptöne hassen.

Time & Scheduling Primitives — "Wann passiert was?"

time.now() → datetime time.is_between(start, end) → bool time.day_of_week() → enum schedule.at(cron_expression, callback) → schedule_id schedule.after(duration, callback) → schedule_id schedule.cancel(schedule_id) → void schedule.every(interval, callback) → schedule_id timer.start(label) → timer_id timer.elapsed(timer_id) → duration

Simpel, aber fundamental. Fast jeder Skill hat eine Zeitkomponente. "Morgens vorlesen", "alle 2 Stunden erinnern", "nach 30 Minuten Inaktivität fragen ob alles okay ist."

State & Memory Primitives — "Was merke ich mir?"

state.set(key, value, ttl?) → void state.get(key) → value | null state.increment(key) → int state.list_push(key, value) → void state.list_recent(key, n) → value[] pattern.track(event_name) → void pattern.frequency(event_name, window) → float pattern.last_occurred(event_name) → datetime | null user_prefs.get(key) → value user_prefs.set(key, value) → void // nur via User-Bestätigung!

Zwei getrennte Systeme: state für Skill-internen Zustand (flüchtig, TTL-basiert), user_prefs für persistente Nutzer-Einstellungen die nur nach expliziter Bestätigung geschrieben werden. Und pattern ist das Subsystem das Phase-0-Trigger-Erkennung ermöglicht — es zählt mit, ohne sofort zu handeln.

LLM / Reasoning Primitives — "Wie denke ich?"

llm.classify(text, categories[]) → category llm.extract(text, schema) → structured_data llm.summarize(text, max_words?) → string llm.generate_response(context, instruction) → string llm.sentiment(text) → positive | neutral | negative | urgent llm.is_similar(text_a, text_b, threshold) → bool llm.translate(text, target_lang) → string

Das sind keine rohen API-Calls. Das sind vorpaketierte, kostenkontrollierte LLM-Operationen mit festen Max-Token-Budgets. Der Coding-Agent darf nie einen open-ended generate mit 100k Tokens auslösen. Jede LLM-Primitive hat ein Budget-Cap, und classify und extract sollten bevorzugt werden weil sie determinisischer und billiger sind als generate_response.

Control Flow Primitives — "Wie verknüpfe ich?"

when(condition, action) → rule_id when_all([conditions], action) → rule_id // AND when_any([conditions], action) → rule_id // OR sequence([actions]) → void // nacheinander gate(condition, action, else_action?) → void cooldown(rule_id, min_interval) → void retry(action, max_attempts, backoff) → result suppress(rule_id, duration) → void

Kein Turing-vollständiges Scripting. Bewusst kein while, kein for, keine Rekursion. Das ist ein regelbasiertes System, kein Programmiersprache. Die Einschränkung ist das Safety-Feature — ein generierter Skill kann keine Endlosschleife produzieren, keine Race Conditions, keinen unkontrollierten Ressourcenverbrauch.

cooldown und suppress sind die eingebauten Nervschutz-Mechanismen. Jede Regel hat automatisch einen Default-Cooldown.

Smart Home / Environment Primitives — "Was steuere ich?"

device.list(type?) → Device[] device.get_state(device_id) → state device.set_state(device_id, state) → result device.toggle(device_id) → result weather.current() → WeatherData weather.forecast(hours) → WeatherData[] calendar.next_event() → Event | null calendar.today() → Event[]

Device-Interaktion über eine einheitliche Abstraktion, nicht direkt gegen Hue/Zigbee/whatever. Der Coding-Agent sagt device.set_state("wohnzimmer_licht", {brightness: 80}) und die Hardware-Schicht darunter löst das auf.

Safety & Meta Primitives — "Wie schütze ich den Nutzer?"

safety.require_confirmation(message) → bool safety.escalate_to_contact(contact_id, reason) → void safety.log(skill_id, event, severity) → void safety.rate_limit_check(skill_id) → bool safety.is_night_mode() → bool skill.disable_self(reason) → void skill.request_feedback(question) → response

Das ist die Schicht die nicht optional ist. require_confirmation für alles was Geld kostet oder nach außen kommuniziert. escalate_to_contact für den Moment wo der Agent merkt dass etwas nicht stimmt. skill.disable_self für den Fall dass ein Skill erkennt dass er nicht richtig funktioniert — eingebauter Kill-Switch.

Und is_night_mode klingt trivial, ist aber essentiell: Nachts um 3 soll kein Skill laut vorlesen oder Lichter anschalten.

Was bewusst NICHT drin ist:

Kein Dateisystem-Zugriff. Kein Network-Socket. Kein Shell-Execute. Kein dynamischer Code-Import. Keine Möglichkeit andere Skills zu modifizieren. Keine direkte Hardware-Ansteuerung. Der Coding-Agent baut innerhalb einer geschlossenen Sandbox — er kann nur kombinieren was exponiert ist, nicht ausbrechen.

Das sind ungefähr 70 Primitives in 8 Kategorien. Aus denen kann der Coding-Agent ziemlich viel bauen — aber nichts Gefährliches. Die Kunst ist: genug Ausdruckskraft für 95% der Alltagswünsche, aber kein Seil um sich aufzuhängen.

Soll ich als nächstes ein paar Beispiel-Skills durchspielen die zeigen wie aus diesen Primitives konkrete Funktionen komponiert werden? Oder willst du erstmal an der Kategorie-Ebene feilen?

Ich meinte eher.. müssten wir nicht sozusagen kleine Scripte/Module bereitstellen, die dann kombiniert werden können? Z.B. ein dcript „Camera.py“ wo wir die Kamera Funktionen haben, eines mit „pir.py“ für Infrarot, eines mit „llm_call.py“ usw?

Mar 15

Erkannte Unterschied zwischen abstrakten APIs und konkreten, wiederverwendbaren Python-Modulen
Erkannte Unterschied zwischen abstrakten APIs und konkreten, wiederverwendbaren Python-Modulen
Ja, genau — und das ist der entscheidende Unterschied. Ich war eine Abstraktionsebene zu hoch. Du meinst die tatsächliche Toolbox die dem Coding-Agent zur Verfügung steht. Nicht ein API-Design, sondern echte, getestete, dokumentierte Module die er importiert und verdrahtet.

Das verändert das Bild fundamental, weil es das Problem massiv vereinfacht. Der Coding-Agent erfindet nichts — er komponiert. Er ist eher ein Elektriker der vorgefertigte Komponenten verdrahtet als ein Ingenieur der Komponenten entwirft.

Lass mich die Module durchgehen wie sie tatsächlich im Dateisystem liegen würden:

Sensor-Module

camera.py — Wrapper um das Kamerasystem. Exponiert Funktionen wie get_frame(), detect_persons(), identify_known_person(frame), is_anyone_visible(). Intern nutzt das ein lokales Modell (YOLO oder MediaPipe), aber der Coding-Agent sieht davon nichts. Er importiert from modules.camera import is_anyone_visible und fertig.

pir.py — PIR-Sensor. Noch simpler: motion_detected(), time_since_last_motion(). Das ist ein GPIO-Wrapper im Kern, aber mit Debouncing und Hysterese schon eingebaut, damit der Coding-Agent sich nicht mit Sensor-Rauschen rumschlagen muss.

microphone.py — is_speech(), listen(timeout) → transkribierter Text, ambient_level(). Das kapselt Whisper oder was auch immer du für STT nutzt. Wichtig: listen() gibt schon Text zurück, nicht Audio-Chunks.

speaker.py — say(text, speed?, volume?), play_sound(sound_name), say_and_wait(text) → User-Antwort als Text. Das letzte ist das mächtigste — ein synchroner Call der spricht, zuhört, transkribiert und den Text zurückgibt.

display.py — Falls twinr ein Display hat: show_text(), show_image(), show_options(list) → selected, clear().

light.py — LED-Ring oder ähnliches: set_color(), pulse(), off(). Für non-verbales Feedback.

Kommunikations-Module

whatsapp.py — get_unread(contact?), send(contact, text), mark_read(msg_id). Dahinter steckt die WhatsApp Business API oder ein Bridge, aber der Coding-Agent sieht nur saubere Funktionen.

sms.py — Gleiche Signatur: get_unread(), send(). Bewusst identisch zu WhatsApp damit der Coding-Agent plattform-agnostisch bauen kann.

email.py — get_unread(filter?), send(to, subject, body), summarize_unread() — wobei summarize_unread schon intern einen LLM-Call macht. Das ist ein Beispiel für ein Modul das selbst schon ein anderes Modul nutzt.

phone.py — call(contact), is_ringing(), get_caller(), answer(), hangup().

contacts.py — find(name_fuzzy), get_all(), get_relationship(contact). Das ist das Adressbuch mit semantischer Suche — "mein Enkel" → findet den richtigen Kontakt.

Denk-Module

llm_call.py — Das zentrale Modul. Aber nicht ein nackter API-Call, sondern vorgefertigte Funktionen: classify(text, options), extract(text, schema), summarize(text), respond(context, instruction), sentiment(text). Jede Funktion hat ein eingebautes Token-Budget. Der Coding-Agent kann keinen 100k-Token-Call auslösen weil es die Funktion dafür schlicht nicht gibt. Und jeder Call loggt automatisch Kosten.

memory.py — remember(key, value, ttl?), recall(key), forget(key), recent(key, n). Persistenter Key-Value-Store für Skill-Zustand. Absichtlich primitiv — kein SQL, keine Queries, einfach Key-Value.

patterns.py — track(event_name), frequency(event_name, days), last_time(event_name), streak(event_name). Das ist das Modul das Phase-0-Trigger-Erkennung ermöglicht — es beobachtet und zählt, ohne zu handeln.

Zeit- und Steuerungs-Module

scheduler.py — every(interval, callback), at(time, callback), after(delay, callback), cancel(job_id). Cron-artig aber mit menschenlesbaren Inputs: every("2 hours"), at("08:00").

rules.py — when(condition_fn, action_fn), when_all([conditions], action), cooldown(rule, interval), suppress(rule, duration). Das ist die Event-Engine. Hier werden Sensor-Module mit Action-Module verdrahtet.

Umgebungs-Module

smarthome.py — devices(), turn_on(device), turn_off(device), set(device, property, value). Unified Interface über Hue, Zigbee, MQTT — was auch immer angebunden ist.

weather.py — now(), forecast(hours), will_rain_today(). Simpel, aber Senioren fragen ständig nach dem Wetter.

calendar.py — today(), next_event(), add(title, time), remind_before(event, minutes).

Safety-Module

safety.py — Das ist das einzige Modul das der Coding-Agent importieren muss. Nicht optional. confirm(question) → bool, escalate(contact, reason), log(event, severity), night_mode() → bool, rate_ok(action_name) → bool.

Und jetzt der Clou: Jedes Modul hat nicht nur Code, sondern ein Docstring-Header der dem Coding-Agent erklärt was es tut, was es braucht, und Beispiele zeigt. So wie du es bei AGENTS.md mit den Skill-Fragmenten machst — der Coding-Agent liest den Header, versteht die Capabilities, und komponiert.

Ein generierter Skill sieht dann so aus:

python
from modules.camera import is_anyone_visible, count_persons
from modules.whatsapp import get_unread
from modules.speaker import say_and_wait
from modules.rules import when_all
from modules.safety import night_mode, confirm

def should_read():
    return (is_anyone_visible() 
            and count_persons() == 1 
            and not night_mode() 
            and len(get_unread()) > 0)

def read_messages():
    for msg in get_unread():
        say_and_wait(f"Nachricht von {msg.sender}: {msg.text}")

when_all([should_read], read_messages, cooldown="5min")
Das ist alles. 15 Zeilen. Kein Boilerplate, kein Setup, keine Hardware-Details. Der Coding-Agent kombiniert Module und liefert ein Skill-Script. Und weil jedes Modul für sich getestet und gehärtet ist, kann auch der generierte Skill nicht viel kaputt machen.

Die Frage die sich dann stellt: wie viele Module brauchst du zum Launch? Ich würde sagen die Kern-15 (camera, pir, microphone, speaker, whatsapp, contacts, llm_call, memory, patterns, scheduler, rules, smarthome, weather, calendar, safety) decken wahrscheinlich 90% der Wünsche ab die eine 78-Jährige haben wird. Und jedes neue Modul das du später hinzufügst erweitert den Möglichkeitsraum ohne bestehende Skills zu brechen.

Willst du eins der Module mal konkret durchspezifizieren — also wirklich die Funktionssignaturen, Error-Handling, Docstrings?

Kannst du das Konzept, die Module usw alles mal in einer Datei zusammenfassen? Also quasi was wollen wir? Warum? Welches Problem lösen wir? Wie setzen wir es konkret um? Module? Flow? Usw.. also von groß nach klein..

Mar 15

Orchestrierte Zusammenfassung von Konzept bis Implementierungsdetails
Orchestrierte Zusammenfassung von Konzept bis Implementierungsdetails
Das Dokument geht von groß nach klein durch: Executive Summary → Problem → Solution Overview → den kompletten 6-Phasen-Flow (Trigger-Erkennung, Feasibility Check, Requirements Gathering, Skill Compilation, Validation & Soft Launch, Monitoring) → den vollständigen Module-Katalog mit allen Funktionssignaturen in 7 Kategorien → ein konkretes Skill-Beispiel → Safety Architecture → TESSAIRACT-Integration → Launch Scope mit MVP-Modulset und Success Metrics → Competitive Positioning.

Intern v0.1 markiert — das ist ein lebendiges Dokument das mit dem Projekt wächst. Wenn du willst können wir als nächstes die einzelnen Module tatsächlich als Python-Stubs mit Docstrings und Typ-Signaturen auskodieren.

Twinr adaptive skill engine
Dokument · DOCX 
Würde es nicht Sinn ergeben, die Codes quasi irgendwie abzuschirmen? So ne Art erzwingender wrapper oder watchdog.. so dass der Agent keine echte sandbox braucht, aber die Scripte gewisse Dinge einfach nicht dürfen? Oder wie würdest du das lösen? Also dass quasi wir nicht nur dem Agent vertrauen, sondern auch harte Garantien erzeugen?

Mar 15

Architektierte mehrschichtige Sicherheitsebenen für Code-Ausführung
Architektierte mehrschichtige Sicherheitsebenen für Code-Ausführung
Genau der richtige Instinkt. "Trust but verify" reicht nicht — du brauchst "don't trust, enforce." Lass mich die Schichten durchgehen, weil die Antwort ist: nicht eine Lösung, sondern vier unabhängige Mauern, von denen jede einzelne allein schon ausreichen würde.

Das Grundproblem: Der Coding Agent ist ein LLM. LLMs halluzinieren, ignorieren Instruktionen, und können durch Prompt Injection manipuliert werden. Wenn die einzige Sicherheit ist "wir haben dem Agent gesagt er soll nichts Böses tun", dann ist die Sicherheit genau null. Du brauchst Garantien die greifen selbst wenn der Agent aktiv versucht auszubrechen.

Schicht 1: AST-Gatekeeper (vor der Ausführung)

Bevor ein generiertes Skill-Script jemals ausgeführt wird, parsed du es als Python AST und walkst den Syntax-Baum. Du hast eine Whitelist erlaubter Konstrukte, und alles was nicht drauf steht wird rejected. Deterministisch, kein LLM, kein Ermessensspielraum.

Erlaubt: import (nur aus modules/), def, if/elif/else, and/or/not, return, Vergleiche, String-Formatting, Tuple/List-Literals, Funktionsaufrufe.

Verboten — und zwar hart, auf AST-Ebene nicht ausdrückbar: while, for, class, eval, exec, compile, __import__, jeder Dunder-Zugriff (__dict__, __globals__, __subclasses__), open, os, sys, subprocess, socket, Lambda (weil man damit for-äquivalente Konstrukte bauen kann), getattr/setattr (weil man damit Sandbox-Escapes bauen kann).

Das ist nicht "wir bitten den Code nett darum" — der AST-Gatekeeper ist ein 200-Zeilen Python-Script das den Parse-Tree durchgeht und bei einem einzigen verbotenen Node das gesamte Skill-Script ablehnt. Kein Fallback, kein "aber es sieht harmlos aus." Ein verbotener Node = rejected.

Schicht 2: Custom Import System (Ladezeit)

Selbst wenn der AST-Gatekeeper einen Edge Case übersieht: Python's Import-System wird überschrieben. Du installierst einen Custom Meta-Path-Finder der physisch nur Module aus dem modules/-Verzeichnis laden kann. Ein import os resolved schlicht zu nichts — nicht zu einem Fehler, sondern zu einem leeren Modul das bei jedem Attributzugriff None zurückgibt und loggt.

Das heißt: selbst wenn jemand einen Weg findet, einen import-Ausdruck am AST-Gatekeeper vorbeizuschmuggeln, gibt es auf Loader-Ebene nichts zu importieren. Die zweite Mauer ist unabhängig von der ersten.

Schicht 3: Restricted Builtins (Laufzeit)

Das Skill-Script läuft in einem exec()-Call mit einem custom Globals-Dict. In diesem Dict sind die Python-Builtins auf eine Whitelist reduziert: len, range, str, int, float, bool, list, dict, True, False, None, print (redirected zum Log). Alles andere existiert nicht.

Das bedeutet: __builtins__ hat kein open, kein eval, kein compile, kein type, kein getattr. Selbst wenn Code es irgendwie an Schicht 1 und 2 vorbeischafft — die Funktionen die er bräuchte um Schaden anzurichten existieren nicht in seiner Laufzeitumgebung.

Schicht 4: OS-Level Enforcement (Prozess-Ebene)

Die letzte Mauer: jeder Skill läuft in einem eigenen Prozess (oder Thread mit seccomp-Filter). Auf Linux-Ebene: seccomp-bpf oder Landlock. Der Prozess darf keine Netzwerk-Syscalls machen, keine Dateien außerhalb eines winzigen tmpfs öffnen, keine Child-Prozesse spawnen, kein ptrace. Das ist Kernel-Level — da kommt kein Python-Trick dran vorbei.

Für twinr auf einem Raspberry Pi oder vergleichbarer Hardware ist seccomp performant genug. Und Landlock (seit Linux 5.13) gibt dir sogar filesystem-level Sandboxing ohne Container-Overhead.

Warum vier Schichten und nicht eine?

Weil Defense-in-Depth das einzige Modell ist das bei unbekannten Angriffsvektoren funktioniert. Jede einzelne Schicht hat theoretische Bypass-Möglichkeiten: AST-Gatekeeper könnte einen Parsing-Edge-Case haben, Custom Imports könnten über importlib Internals umgangen werden, Restricted Builtins haben historisch Escapes über object.__subclasses__(), und seccomp-Filter können zu permissiv konfiguriert sein.

Aber alle vier gleichzeitig zu bypassen ist praktisch unmöglich. Ein Angriff müsste einen AST-Konstrukt finden der gleichzeitig durch den Parser rutscht, ein Modul laden das der Custom Loader nicht blockt, eine Builtin-Funktion aufrufen die nicht existiert, und dann noch einen Syscall machen den seccomp durchlässt. Das ist keine realistische Bedrohung.

Plus ein fünftes Element: der Watchdog

Neben den vier Enforcement-Schichten läuft ein Watchdog-Prozess der zur Laufzeit Metriken überwacht: CPU-Zeit pro Skill-Execution (Kill nach 5 Sekunden), Memory-Verbrauch (Kill nach 50MB), Anzahl LLM-Calls pro Minute (Rate-Limit in den Modulen selbst), Anzahl externer Kommunikations-Events (max N pro Stunde). Der Watchdog hat die Autorität, jeden Skill sofort zu terminieren und zu deaktivieren.

Wie das in der Praxis aussieht:

Der Coding Agent liefert ein Script ab. Dann passiert:

Schritt 1 — AST-Gatekeeper parsed und validiert. Grün oder Reject.
Schritt 2 — Script wird mit Custom Import Loader und Restricted Builtins in einem seccomp-gefilterten Subprocess ausgeführt.
Schritt 3 — Watchdog überwacht die Ausführung.
Schritt 4 — Tests laufen im selben Sandbox-Modus.
Schritt 5 — Erst nach bestandenen Tests geht der Skill in den Produktiv-Scheduler, wo er bei jedem Aufruf denselben Sandbox-Prozess durchläuft.

Der Overhead ist minimal — wir reden über 15-Zeilen-Scripts die Module-Funktionen aufrufen. Die eigentliche Arbeit passiert in den Modulen, die außerhalb der Sandbox im vertrauenswürdigen Hauptprozess laufen. Die Sandbox schützt nur die Verdrahtungsebene.

Der ehrliche Punkt: Schicht 4 (seccomp/Landlock) ist der Engineering-Aufwand. Die Schichten 1-3 sind jeweils ein halber Tag Arbeit. Seccomp-Profile richtig zu konfigurieren dass sie tight genug sind um zu schützen, aber loose genug dass die Module noch funktionieren, das braucht sorgfältiges Testing. Aber es ist ein einmaliger Aufwand, kein wiederkehrender.

Soll ich das ins Konzeptdokument als neues Kapitel "Enforcement Architecture" einbauen, oder willst du erstmal den AST-Gatekeeper als konkreten Prototypen sehen?

Wäre gut, ja.. wir dürfen aber auch nicht so restriktiv sein, dass am Ende nur Scripte möglich sind, die am Ende einfach nur ein if than sind.. ich denke schon dass komplexere Wünsche möglich sein sollten, sowas wie: der skill soll schauen, welche meiner Mails von Leuten kommen, die ich auch in WhatsApp habe, und mir nur diese vorlesen.. aber wenn eine Mail nur eine Info will zu meinem .. keine Ahnung, Terminplan morgen, dann lies die nicht vor, sondern schick meinen Terminplan..

Mar 15

Reconsidered Sicherheitsrestriktionen zugunsten komplexerer Skill-Logik
Reconsidered Sicherheitsrestriktionen zugunsten komplexerer Skill-Logik
Guter Punkt — und er zeigt ein echtes Problem in meinem bisherigen Framing. "Kein for, kein while" war zu restriktiv. Dein Beispiel braucht Iteration, Cross-Module-Joins, und LLM-gesteuerte Entscheidungsbäume. Das ist kein simples If-Then mehr.

Lass mich das Enforcement-Modell richtig aufsetzen und dann ins Dokument einbauen.

Der Paradigmen-Shift ist: wir schränken nicht die Logik-Komplexität ein, sondern den Zugriff auf gefährliche Ressourcen. Ein Skill darf beliebig clever sein — er darf nur nicht aus der Box raus.

Konkret heißt das: for x in collection ist erlaubt — es iteriert über eine endliche Liste die ein Modul zurückgibt. while True bleibt verboten — das ist potentiell unendlich. Und der Watchdog ist das Sicherheitsnetz: selbst wenn ein for-Loop über eine unerwartet riesige Liste läuft, killt der Watchdog nach 5 Sekunden.

Dein Beispiel als Skill würde so aussehen:

python
from modules.email import get_unread
from modules.whatsapp import get_contacts
from modules.calendar import tomorrow
from modules.llm_call import classify, extract
from modules.speaker import say
from modules.whatsapp import send
from modules.safety import night_mode, log

def check_emails():
    wa_contacts = get_contacts()
    wa_emails = {c.email for c in wa_contacts if c.email}
    
    for mail in get_unread():
        if mail.sender not in wa_emails:
            continue
        
        intent = classify(
            mail.body,
            ["wants_information", "shares_information", "action_required"]
        )
        
        if intent == "wants_information":
            topic = extract(mail.body, {"requested_info": "string"})
            if "termin" in topic["requested_info"].lower():
                schedule = tomorrow()
                send(mail.sender_contact, f"Mein Terminplan: {schedule}")
                log("auto_reply_schedule", "info")
        
        elif intent == "shares_information":
            say(f"Mail von {mail.sender_name}: {mail.subject}")
Das ist komplex, cross-modular, LLM-gesteuert — und trotzdem sicher, weil jeder einzelne Funktionsaufruf in ein gehärtetes Modul geht.

Ich bau jetzt das Enforcement-Kapitel ins Dokument ein — mit dem korrigierten Modell.


Reorganisierte Dokumentation mit erweiterten Beispielen und detaillierter Architektur
Reorganisierte Dokumentation mit erweiterten Beispielen und detaillierter Architektur
Now I'll make the edits — replacing the simple example with two examples (simple + complex), and replacing the old Safety chapter with a proper Enforcement Architecture.


Reorganisierte Sicherheitsarchitektur-Kapitel strukturell um
Reorganisierte Sicherheitsarchitektur-Kapitel strukturell um
Now let me replace the Safety Architecture section with the full Enforcement Architecture:


Verifizierte Kapitelstruktur und bestätigte unveränderte Nummerierung
Verifizierte Kapitelstruktur und bestätigte unveränderte Nummerierung
Now let me rebuild and validate:


Entwarf flexible Automatisierungslösung für intelligente E-Mail-Verwaltung
Entwarf flexible Automatisierungslösung für intelligente E-Mail-Verwaltung
Das Dokument hat jetzt drei wesentliche Änderungen:

Kapitel 6 zeigt zwei Beispiele — den einfachen WhatsApp-Read-Aloud (15 Zeilen) und deinen komplexen Email-Router mit Cross-Channel-Join, LLM-Klassifikation und Auto-Reply (40+ Zeilen). Das setzt die Messlatte für das, was das System können muss.

Kapitel 7 ist komplett neu: die vierschichtige Enforcement Architecture (AST Gatekeeper → Custom Import System → Restricted Builtins → seccomp/Landlock) plus Watchdog. Die Kernidee: for x in collection ist erlaubt (bounded iteration), while bleibt verboten (unbounded). Logische Komplexität ist uneingeschränkt, Ressourcenzugriff ist komplett kontrolliert.

rules.py Beschreibung ist korrigiert — kein "deliberately not Turing-complete" mehr, stattdessen der Fokus auf Cooldown/Suppress als Überlastschutz.

Das Balanceprinzip in einem Satz: der Smart Email Router aus 6.2 läuft problemlos durch alle vier Enforcement-Layer, aber ein import os; os.system("rm -rf /") scheitert an jeder einzelnen Schicht unabhängig.

Twinr adaptive skill engine
Dokument · DOCX 
Was hiervon ist sinnvoll und was übertrieben?

Ja. Das ist jetzt nahe an einer echten Sicherheitsarchitektur.

Die zentrale Korrektur ist diese:

AST-Filter, Custom-Import und Restricted-Builtins sind gute Vorfilter. Die echten Sicherheitsgrenzen sind nur:
    1.    separater Prozess,
    2.    Kernel-Enforcement,
    3.    ein versiegelter Capability-/Policy-Broker.

Alles andere ist wichtig, aber nicht die letzte Mauer.

AST-Gatekeeper: ja, aber als strikt version-gepinntes Subset.
Das ist richtig. Der Parser muss auf genau die Python-Version gepinnt sein, die du in Produktion fährst, weil sich die AST-Grammatik zwischen Releases ändern kann. Und du musst nicht nur for/while/lambda bannen, sondern auch die versteckten Iterations- und Kontrollflussformen: ListComp, SetComp, DictComp, GeneratorExp, NamedExpr, Match, Try, With, Async*, Yield*, Decorators, Klassen, sowie praktisch jeden allgemeinen Attribute-Zugriff. Die Python-Doku zeigt explizit, dass diese Knoten Teil der AST-Grammatik sind und dass die abstrakte Syntax sich pro Release ändern kann.  ￼

Der Import-Layer muss fail-closed sein. Kein Dummy-Modul.
Ein blockierter Import darf nicht in ein leeres Objekt degenerieren, das bei Attributzugriff None liefert. Das ist kein Sicherheitsgewinn, sondern semantische Korruption und Debugging-Hölle. Der saubere Move ist härter: keine Runtime-Imports im Skill-Prozess. Lass den Trusted Parent die Imports aus dem AST extrahieren, gegen eine Allowlist auflösen und dann die erlaubten Capability-Namen direkt in das Sandbox-Globals-Dict injizieren. Das ist robuster, weil das import-Statement in Python über __import__() läuft und die Import-Maschinerie selbst stark anpassbar ist über sys.meta_path, sys.path_hooks und sys.path_importer_cache. Je weniger davon du im untrusted Prozess überhaupt benutzt, desto besser.  ￼

Restricted Builtins sind nützlich, aber ausdrücklich keine Sicherheitsgrenze.
Hier ist Python selbst sehr klar: Bei exec()/eval() wird __builtins__ automatisch eingefügt, wenn es fehlt, und das Überschreiben von __builtins__ ist laut offizieller Doku keine Sicherheitsmechanik. Dazu kommt ein noch wichtigerer Punkt: normale Python-Funktionsobjekte tragen function.__builtins__ und function.__globals__ mit sich. Deshalb darfst du der Sandbox nie echte Python-Modulobjekte oder normale Python-Funktionen aus dem trusted Prozess geben und dann hoffen, dass AST-Filter plus Builtins-Whitelist reichen. Gib ihr stattdessen nur opake Capability-Proxies oder RPC-Stubs. Dann ist der untrusted Code nie im Besitz echter Interpreter-intern relevanter Objekte.  ￼

Attributzugriff ist das eigentliche Minenfeld.
Wenn du msg.sender erlaubst, öffnest du die Tür zu einem großen Teil des Python-Objektmodells. Für ein hartes Subset würde ich daher Skills entweder ganz ohne allgemeinen Attributzugriff fahren oder ihn nur auf inert-formatige Records zulassen. Praktisch: lieber msg["sender"] auf einem versiegelten Mapping oder dedizierte Accessor-Funktionen als freie Attribut-Navigation. Sonst bist du sehr schnell wieder bei __class__, __dict__, __globals__, __builtins__ und angrenzenden Escape-Flächen. Die Python-Doku zeigt, wie reichhaltig genau diese Objekt- und Modul-Namespace-Flächen sind.  ￼

OS-Level: das ist die erste wirklich harte Mauer.
Hier ist dein Instinkt richtig. Linux trennt die Rollen sogar explizit: Landlock ist für Objekt-/Zugriffsrestriktion gedacht, seccomp für Syscall-Filtering. Beides zusammen ist sinnvoll. Aber: nimm Prozesse, nicht Threads. Die seccomp-Manpage sagt explizit, dass SECCOMP_RET_KILL_THREAD in einem multithreaded Prozess diesen in einen permanent inkonsistenten oder korrupten Zustand bringen kann. Prozess-pro-Skill ist deshalb der richtige Grundschnitt.  ￼

Bei Landlock gibt es zwei praktische Fallstricke, die du im Design fest einbauen musst.
Erstens: Dateien oder Verzeichnisse, die vor dem Sandboxing geöffnet wurden, sind von den Landlock-Restriktionen nicht mehr betroffen; Rechte hängen teils am File Descriptor. Zweitens: Landlock-Features sind ABI-abhängig. TCP-Connect/Bind kam erst mit ABI v4, einige IPC-Scopes noch später, und Thread-Synchronisierung über alle Threads des Prozesses sogar erst mit ABI v8. Daraus folgt: Sandbox-Prozess mit sauberem FD-Table starten, alles Unnötige schließen, nur exakt definierte IPC-Kanäle vererben, und Landlock-ABI zur Laufzeit prüfen.  ￼

Die wichtigste zusätzliche Mauer fehlt noch in deiner Aufzählung: der Capability-/Effect-Broker.
Der Skill darf nie selbst entscheiden, ob er safety.confirm() vorher aufgerufen hat. Das wäre wieder Vertrauen in untrusted Code. Stattdessen muss jede effectful Capability durch einen trusted Broker laufen, der zur Laufzeit Kontext und Policy prüft: Privacy-Klasse, Nutzerzustimmung, Nachtmodus, Anwesenheit Dritter, Rate-Limits, Cooldowns, Beziehungsmodell, Skill-Risikoklasse.
Beispiel:
whatsapp.send() ohne Broker-Grant → hart denied.
speaker.say_private_message() bei others_present=True → hart denied.
phone.call() ohne explicit approval token → hart denied.
Das ist die eigentliche Übersetzung von „don’t trust, enforce“.

Der Watchdog ist richtig, aber nur als Availability-Layer.
CPU-Zeit, Wall-Clock, RSS, IPC-Volumen, maximale Zahl effectful Calls, maximale String-/Payload-Größe, maximale Zahl Capability-Invocations pro Run. Kill und disable bei Überschreitung. Aber der Watchdog ist nicht die primäre Sicherheitsgrenze, sondern der Airbag.

Die saubere Endform wäre deshalb:

A. Generated Python
B. AST-Validator + Import-Extraktion + Effect-Manifest
C. Trusted Loader bindet nur erlaubte Capability-Proxies
D. Ausführung in separatem Worker-Prozess mit no_new_privs + seccomp + Landlock
E. jeder Capability-Call geht über einen trusted Policy-Broker
F. Watchdog killt bei Budgetverletzung
G. Audit-Log auf jeder Deny-/Kill-/Policy-Entscheidung

Der wichtigste einzelne Satz dafür lautet:

Untrusted Python may compose behavior; only trusted code may authorize effects.

Das ist der Punkt, an dem das Design von „klug“ zu „tragfähig“ kippt.

Ich würde daraus im Dokument eine eigene Sicherheitssektion machen mit genau drei Begriffen:
sealed capability surface · fail-closed execution · brokered effects.

Mehr anzeigen
Mar 15

Analysierte technisches Feedback und identifizierte Kernarchitektur-Prinzipien
Analysierte technisches Feedback und identifizierte Kernarchitektur-Prinzipien
Das ist fast alles sinnvoll — und an den entscheidenden Stellen besser als mein ursprüngliches Design. Lass mich ehrlich sortieren:

Vollständig richtig und wichtig:

Punkt 7 (Capability-Broker) ist der größte Treffer. Das fehlte komplett. Mein Design hatte safety.confirm() als Funktion die der Skill freiwillig aufruft — das ist exakt so sicher wie eine Tür die sich selbst abschließen soll. Der Broker-Pattern dreht das um: kein Effect ohne Grant vom Trusted Process. Das ist der architektonische Unterschied zwischen "wir bitten den Code nett" und "wir erzwingen es."

Punkt 2 (keine Runtime-Imports, sondern Injection durch Trusted Parent) ist ebenfalls ein klarer Upgrade. Meine Idee mit dem "Dummy-Modul das None zurückgibt" war tatsächlich semantische Korruption. Der saubere Weg — AST-Imports extrahieren, gegen Allowlist prüfen, Capability-Proxies ins Globals-Dict injizieren — eliminiert die gesamte Import-Maschinerie aus dem untrusted Prozess.

Punkt 3 (Restricted Builtins sind kein Sicherheitsmechanismus) ist korrekt und ich hatte das falsch geframed. __builtins__ Manipulation ist laut Python-Docs explizit kein Security-Feature. Und der Punkt mit function.__globals__ und function.__builtins__ ist der Killer: wenn du echte Funktionsobjekte aus dem Trusted Process in die Sandbox gibst, kannst du über deren Attribute ausbrechen. Daher: nur opake Proxies, nie echte Python-Objekte.

Punkt 4 (Attributzugriff als Minenfeld) ist richtig. msg.sender → msg.__class__.__bases__[0].__subclasses__() ist ein klassischer Escape. Die Lösung msg["sender"] auf versiegelten Mappings oder Accessor-Funktionen ist sauberer. Etwas weniger ergonomisch für den Coding Agent, aber das ist ein akzeptabler Trade-off.

Punkt 5 und 6 (Prozesse statt Threads, Landlock ABI-Versioning, FD-Hygiene) sind wichtige Praxis-Details die in der Implementierung über Erfolg oder Scheitern entscheiden.

Der eine Punkt der leicht übertrieben ist:

Punkt 1 (AST version-pinning) ist technisch korrekt, aber in der Praxis weniger kritisch als dargestellt. Ja, die AST-Grammatik ändert sich zwischen Python-Releases. Aber twinr läuft auf embedded Hardware mit einer kontrollierten Python-Version. Das ist keine Web-App die auf beliebigen Python-Versionen deployt wird. Version-Pinning ist trotzdem richtig als Praxis, aber es ist kein realistischer Angriffsvektor.

Die endgültige Drei-Wort-Architektur ist perfekt:

Sealed Capability Surface · Fail-Closed Execution · Brokered Effects.

Das ist besser als mein "vier unabhängige Mauern"-Framing, weil es die Hierarchie richtig darstellt: die Python-Level-Filter (AST, Imports, Builtins) sind Defense-in-Depth Pre-Filter, aber die echten Sicherheitsgrenzen sind der isolierte Prozess, die Kernel-Enforcement, und der Capability-Broker.

Ich bau das jetzt um.


Identified notwendige Codeänderung für Dictionary-Zugriff statt Attributzugriff
Identified notwendige Codeänderung für Dictionary-Zugriff statt Attributzugriff
Now I need to update the skill examples to use dict-style access (msg["sender"] instead of msg.sender) since the new architecture forbids Attribute nodes:


Überarbeitete Dokumentation zur aktualisierten Sicherheitsarchitektur und Broker-Struktur
Überarbeitete Dokumentation zur aktualisierten Sicherheitsarchitektur und Broker-Struktur
Now update the commentary after the complex example and the safety module description:


Synthesized comprehensive security architecture critique and refinements
Synthesized comprehensive security architecture critique and refinements
Die wesentlichen Änderungen im Überblick:

Kapitel 7 komplett umgebaut — von "vier gleichwertige Mauern" zu einer klaren Hierarchie mit den drei Pfeilern Sealed Capability Surface · Fail-Closed Execution · Brokered Effects.

Die Python-Level-Filter (AST, Imports, Builtins) sind jetzt ehrlich als Tier 1: Defense-in-Depth deklariert — nützlich, aber explizit keine Sicherheitsgrenze. Die echten Grenzen sind Tier 2 (Prozess-Isolation, seccomp, Landlock mit FD-Hygiene und ABI-Versioning) und Tier 3 (der Capability Broker).

Der Broker ist das Herzstück. safety.confirm() als freiwillige Funktion existiert nicht mehr. Stattdessen: jeder Effect ist ein RPC-Call an den Trusted Parent, der durch den Broker läuft. Die Policy-Dimensions-Tabelle (Privacy, Consent, Time, Rate, Risk Class, Relationship, Cost) zeigt was der Broker bei jeder Anfrage prüft.

Skill-Code nutzt jetzt dict-Zugriff (msg["sender"] statt msg.sender) — Attribute-Nodes sind im AST verboten, weil sie die __class__.__subclasses__()-Escape-Kette öffnen. Etwas weniger hübsch, aber sicher.

Das safety.py Modul ist auf read-only Kontextabfragen reduziert (night_mode, log, disable_skill). Alles Effectful läuft über den Broker.

Der zentrale Satz steht jetzt kursiv im Dokument: Untrusted Python may compose behavior; only trusted code may authorize effects.

Twinr adaptive skill engine
Dokument · DOCX 
Kannst du das als Markdown ausgeben, und zwar so, dass es ein Coding-Agent dann umsetzen kann?

12:38 PM

Strukturierte Implementierungsvorlage in Markdown verfasst
Strukturierte Implementierungsvorlage in Markdown verfasst
~1.300 Zeilen, ~64KB. Die wesentlichen Unterschiede zum docx-Konzeptdokument:

Jede Phase hat jetzt einen "Implementation Notes"-Block mit konkreten Anweisungen: welche Datenstrukturen, welche Funktionsaufrufe, welche State-Machine-Zustände.

Die Module sind als vollständige Python-Stubs mit Docstrings und Typ-Signaturen ausgeschrieben — ein Coding Agent kann daraus direkt die Modul-Skelette generieren.

Die Enforcement-Architektur enthält lauffähige Code-Skelette für den AST-Validator, den Trusted Loader mit Import-Extraktion und RPC-Proxy-Generierung, den Skill-Runner mit Fork/Sandbox-Aufbau, den Capability Broker mit Policy-Evaluation, und das Landlock-Setup.

Plus ein ASCII-Architekturdiagramm das die Prozess-Grenzen, IPC-Pipes und Trust-Boundaries auf einen Blick zeigt.

Das SkillSpec-Schema (Output von Phase 2, Input für Phase 3) ist als konkretes Dict-Schema definiert — damit weiß der Coding Agent exakt was er als Input bekommt.

Twinr adaptive skill engine
Dokument · MD 


Möchtest du benachrichtigt werden, wenn Claude antwortet?



Claude ist eine KI und kann Fehler machen. Bitte überprüfe die Antworten.
Twinr adaptive skill engine · MD
Kopieren

# twinr Adaptive Skill Engine (ASE)
 
> **Architecture & Implementation Specification**
> Arculae / twinr Project · Thom Heinrich · March 2026 · Internal Draft v0.1
> **CONFIDENTIAL**
 
---
 
## 1. Executive Summary
 
twinr is a physical AI companion for senior citizens — a voice-first device that helps elderly people manage their daily lives independently. Unlike existing assistants (Alexa, Google Home), twinr understands its user personally and adapts over time.
 
This document specifies the **Adaptive Skill Engine (ASE)**: a runtime subsystem that allows twinr to learn new capabilities at runtime, driven by the user's needs. When a user asks for something twinr cannot yet do, the system proposes to learn it, conducts lightweight requirements gathering, delegates code generation to a background coding agent, validates the result, and deploys the new skill — all without any technical expertise from the user.
 
The core UX promise is: **"I can't do that yet, but I can learn it. Give me a few minutes."**
 
This transforms twinr from a static assistant with a fixed feature set into a living system that shapes itself around its user — the opposite of forcing seniors to adapt to technology.
 
---
 
## 2. The Problem
 
### 2.1 Why Existing Assistants Fail Seniors
 
Voice assistants are built for the average user, not for seniors. The failure modes are specific and compounding:
 
- **Fixed capabilities:** Alexa can only do what Amazon ships. If a senior needs a slightly different workflow ("read me my WhatsApp messages but only from family"), the answer is "I can't do that." That's a dead end.
- **Skills/Actions require installation:** Alexa Skills and Google Actions require the user (or a relative) to find, enable, and configure them. Seniors rarely do this. The feature might as well not exist.
- **No personalization:** Assistants don't learn their user's patterns, preferences, or routines. They treat every interaction as if it's the first.
- **No graceful degradation:** When something fails, the user gets a cryptic error or silence. There is no recovery path that doesn't involve a technically competent person.
 
### 2.2 The Core Insight
 
Seniors don't need a better assistant. They need an assistant that gets better — that adapts to them rather than requiring them to adapt to it. The Adaptive Skill Engine is the mechanism that makes this possible.
 
---
 
## 3. Solution Overview
 
### 3.1 The Adaptive Skill Engine (ASE)
 
The ASE is a runtime subsystem with three core components:
 
1. **The Conversational Agent** (front-stage): The voice-first agent that the user interacts with. It detects unmet needs, proposes learning, conducts requirements gathering, and later executes the learned skills.
2. **The Coding Agent** (back-stage): A background process (powered by Claude Code, Codex, or equivalent) that receives a structured skill specification and composes a working skill from pre-built, tested modules. It also generates tests and validates the result.
3. **The Module Library** (toolbox): A curated set of Python modules that expose every hardware capability, communication channel, LLM function, and smart home integration as simple, documented, sandboxed functions. The Coding Agent can only import from this library — it cannot access raw hardware, the filesystem, or the network directly.
 
### 3.2 Key Architectural Principle: Composition, Not Creation
 
The Coding Agent does not write code from scratch. It composes pre-existing, pre-tested building blocks. This is analogous to an electrician wiring a house: the components (switches, outlets, breakers) are manufactured and certified; the electrician's job is to connect them correctly for the specific house.
 
This dramatically reduces the risk surface. Generated skills cannot access anything outside the module sandbox, cannot create infinite loops (the AST validator rejects `while`), and cannot exceed resource budgets (every LLM call has a built-in token cap).
 
---
 
## 4. End-to-End Flow
 
### 4.1 Phase 0: Trigger Detection
 
The system must recognize when a new skill is needed. Three trigger classes exist:
 
**Explicit Request.** The user directly asks for a capability: "Can you always tell me when I get a new message from Janina?" The conversational agent recognizes this as a skill request via intent classification.
 
**Implicit Frustration.** The user repeatedly asks for something the agent cannot do. Example: asking "Do I have new messages?" three times across two days. The `patterns` module tracks event frequency and surfaces this as a candidate. The agent then proposes proactively: "You ask me that often — want me to learn how to check automatically?"
 
**Observed Routine.** The agent detects a recurring behavioral pattern: every morning at 8, the user turns on the light, asks for weather, and checks the calendar. After several days of consistent pattern detection, the agent offers: "Shall I do your morning routine automatically when you get up?" This is the most powerful trigger but also the most invasive — a high confidence threshold is required before proposing.
 
**Suppression Logic:** Not every gap should trigger a proposal. A global heuristic filters candidates: `frequency × emotional_relevance × technical_feasibility`. Maximum one learning proposal per day to avoid overwhelming the user. Once-off requests (e.g., a single question about stock prices) are not candidates.
 
#### Implementation Notes (Phase 0)
 
- The `patterns.py` module (see Section 5) provides `track()`, `frequency()`, `last_time()`, and `streak()` — these are the data sources for trigger detection.
- Trigger classification runs as a background task in the Conversational Agent, not as a user-facing skill.
- The suppression heuristic must be tunable per-user (some seniors are more receptive to proactive suggestions than others).
- Store trigger candidates in `memory.py` with a TTL of 7 days.
 
### 4.2 Phase 1: Feasibility Check
 
Before making any promise to the user, the agent must verify internally whether the requested skill is buildable. This is a **deterministic check**, not an LLM guess.
 
The agent constructs a preliminary skill spec from the recognized intent and checks it against the **Capability Registry**:
 
- Are all required modules available and configured? (e.g., is WhatsApp connected?)
- Is the combination of modules permitted by the safety policy?
- Do similar skills already exist in the Skill Library that can serve as templates?
 
Three possible outcomes:
 
| Outcome | Condition | User-Facing Response |
|---------|-----------|---------------------|
| ✅ Green | All capabilities present, similar skills exist | "I can learn that. Give me a few minutes." |
| ⚠️ Yellow | Capabilities present but novel combination | "I think I can learn that, but I'm not fully sure. Want me to try?" |
| ❌ Red | Missing capability (e.g., no WhatsApp access) | "I can't do that right now because I don't have access to your WhatsApp. Your grandson could set that up — shall I send him a message?" |
 
The Red path **never leads to a dead end**. It always offers an escalation path to a trusted contact person.
 
#### Implementation Notes (Phase 1)
 
- The Capability Registry is a JSON/YAML manifest listing every installed module, its status (configured/unconfigured), and the capabilities it exposes.
- The feasibility checker is a deterministic function: `check_feasibility(skill_spec) → Green | Yellow | Red`.
- For Yellow results: track the Coding Agent's success/failure on novel combinations to improve future feasibility predictions.
- For Red results: maintain a `trusted_contacts` list with per-contact capabilities (e.g., "grandson can configure tech", "daughter handles medical").
 
### 4.3 Phase 2: Requirements Gathering
 
If the feasibility check passes (Green or Yellow), the agent transitions into a brief, structured requirements conversation. The mode switch is announced explicitly:
 
> "Okay, I'd like to learn that. I need to ask you two or three short questions so I get it right. Alright?"
 
**Hard constraint: Maximum 3 questions per skill.** If the agent needs more than 3 clarifications, the skill spec is too complex and must be internally decomposed into multiple simpler skills.
 
Questions follow a fixed pattern:
 
| Dimension | Example Question | Purpose |
|-----------|-----------------|---------|
| **When** (Trigger) | "Should I do this automatically, or only when you ask?" | Determines push vs. pull behavior |
| **What** (Scope) | "All WhatsApp messages, or only from specific people?" | Defines the data filter |
| **How** (Modality) | "Should I just read it out, or ask you first?" | Sets interaction pattern |
 
Every question has **sensible defaults**. If the user says "Just do whatever makes sense," the agent picks the default and confirms: "Okay, I'll read out all new messages when you're nearby. Sound good?"
 
**Interpreting ambiguous answers:** When the user adds a condition in natural language ("but not when I have visitors"), the agent uses an LLM call to extract a structured constraint (`trigger_condition += no_other_persons_visible`). If interpretation is uncertain, the agent mirrors back: "Just to make sure — you mean I should only read messages when you're alone in the room?"
 
#### Implementation Notes (Phase 2)
 
- The requirements dialogue is a state machine with states: `ANNOUNCE → ASK_WHEN → ASK_WHAT → ASK_HOW → CONFIRM → DONE`.
- Each state has a default transition if the user gives a non-answer ("I don't know", "Whatever", "Just do it").
- Ambiguous answer interpretation uses `llm_call.extract(user_response, constraint_schema)` where `constraint_schema` defines the structured output format.
- The mirror-back pattern ("Just to make sure...") is triggered when `llm_call.classify(user_response, ["clear", "ambiguous"])` returns "ambiguous".
- Output of Phase 2 is a complete `SkillSpec` object (see below).
 
```python
# SkillSpec schema — output of Phase 2, input to Phase 3
SkillSpec = {
    "name": str,              # human-readable skill name
    "trigger": {
        "type": "push" | "pull",
        "conditions": list[str],  # e.g. ["user_visible", "new_whatsapp", "not_others_present"]
    },
    "action": str,            # primary action description
    "scope": dict,            # filters (contacts, platforms, categories)
    "constraints": list[str], # e.g. ["not_during_night_mode"]
    "capabilities": list[str] # required module names
}
```
 
### 4.4 Phase 3: Skill Compilation
 
After requirements gathering, the conversational agent has a complete, structured skill specification:
 
```
Skill:        whatsapp_read_aloud
Trigger:      user_visible AND new_whatsapp AND NOT others_present
Action:       read_message_aloud(sender, content)
Scope:        all_contacts
Mode:         push (automatic)
Constraints:  not_during_night_mode
Capabilities: [camera, whatsapp, speaker, safety]
```
 
This spec is handed to the Coding Agent (a background process), which:
 
1. Selects the required modules from the Module Library.
2. Reads each module's docstring header to understand available functions.
3. Composes a skill script that imports and wires the modules.
4. Generates test cases for the skill (happy path + edge cases).
5. Runs the tests in the enforcement sandbox (see Section 7).
6. Optionally: a second LLM pass reviews the generated code for safety issues.
 
The user is not involved during this phase. The agent simply says: "I'm learning that now. I'll let you know when I'm ready."
 
Expected compilation time: 2–90 minutes depending on complexity and whether similar skills exist as templates.
 
#### Implementation Notes (Phase 3)
 
- The Coding Agent receives: the SkillSpec, the docstring headers of all required modules, and up to 3 example skills from the Skill Library that use similar module combinations.
- The Coding Agent's system prompt must enforce: dict-style data access only (`msg["sender"]`, not `msg.sender`), no `while`/`class`/`lambda`/`try`, all effects go through module functions (never direct I/O).
- The generated skill script must include a metadata header comment with: skill name, compilation timestamp, spec hash, required capabilities.
- Test generation: at minimum, one happy-path test, one edge-case test (e.g., empty inbox), and one negative test (trigger conditions not met → skill does nothing).
- All tests run inside the full enforcement sandbox (Section 7).
 
### 4.5 Phase 4: Validation & Soft Launch
 
When compilation succeeds, the agent does not silently activate the skill. Instead, it initiates a guided test:
 
> "I've learned it! Want to try it out? Send yourself a message and I'll show you how it works."
 
This serves three purposes: it lets the user verify correct behavior, it builds trust ("I can see what it does before it runs on its own"), and it catches edge cases that automated testing might miss (e.g., the user's WhatsApp display name differs from their contact name).
 
Only after the user confirms the test does the skill enter production mode.
 
#### Implementation Notes (Phase 4)
 
- The guided test runs the skill in a "dry run" mode where the Capability Broker logs all effect requests but also executes them (unlike test mode where effects are mocked).
- The user confirmation is binary: "Yes, that's right" → activate. "No, that's not what I meant" → loop back to Phase 2.
- If the user says "almost right, but..." → partial re-spec, then Phase 3 re-compilation with the delta.
 
### 4.6 Phase 5: Monitoring & Feedback Loop
 
After 2–3 days of active use, the agent checks in:
 
> "I've been reading your messages out for a few days now — is that working well, or should I change something?"
 
If the user reports issues or wants adjustments, the system loops back to Phase 2 (requirements refinement) or Phase 3 (recompilation). Skills can also be disabled at any time: "Stop reading my messages" → skill paused, not deleted, with option to reactivate.
 
Additionally, each skill tracks its own health metrics: error rate, trigger frequency, user interruption rate. A skill that triggers often but gets interrupted frequently auto-flags for review.
 
#### Implementation Notes (Phase 5)
 
- Health metrics are stored via `memory.py` with keys like `skill:{skill_id}:error_count`, `skill:{skill_id}:trigger_count`, `skill:{skill_id}:interrupt_count`.
- The feedback check-in is scheduled by `scheduler.at()` for 72 hours after skill activation.
- Skill states: `active`, `paused` (user-initiated), `disabled` (watchdog-initiated), `retired` (user deleted).
- Skill versioning: when a skill is recompiled, the previous version is retained for rollback. Maximum 3 versions per skill.
 
---
 
## 5. The Module Library
 
The Module Library is the toolbox from which the Coding Agent builds skills. Every module is a standalone Python file with a clear docstring header, typed function signatures, built-in error handling, and no external dependencies beyond what twinr's runtime provides.
 
The Coding Agent cannot import anything outside this library. This is the fundamental security boundary.
 
### 5.1 Design Principles
 
- **Atomic functions:** Each function does exactly one thing. No function has side effects beyond its documented purpose.
- **Semantic abstraction:** Modules expose the lowest useful abstraction, not raw hardware. `camera.py` returns "person detected", not pixel arrays.
- **Fail-safe defaults:** Every function returns a safe default on error (`False`, empty list, `None`) rather than throwing exceptions. Skills built from these modules cannot crash.
- **Built-in budgets:** LLM calls have hardcoded token caps. Scheduling functions have rate limits. No module allows unbounded resource consumption.
- **Self-documenting:** Each module's docstring header serves as the Coding Agent's API documentation. The header includes function signatures, parameter descriptions, return types, and usage examples.
- **Dict-style returns:** All data returned from modules uses sealed dict format (`msg["sender"]`, not `msg.sender`) to prevent Python object-model escape chains. See Section 7.2.3.
 
### 5.2 Module Catalog
 
Every module listed below must be implemented as a Python file in the `modules/` directory. Each module must include a docstring header that the Coding Agent reads before composing skills. The functions listed are the **public API** — the Coding Agent sees only these. Internal implementation details are hidden.
 
**Important:** In the enforcement architecture (Section 7), these modules run in the **trusted parent process**. The skill process receives only RPC proxy stubs for each function. The module code itself is never exposed to untrusted skill code.
 
#### 5.2.1 Sensor Modules
 
**`camera.py`**
 
```python
"""
Camera input with built-in person detection.
Returns semantic results, never raw frames to skill code.
Backed by YOLO/MediaPipe running locally.
"""
 
def is_anyone_visible() -> bool:
    """Returns True if at least one person is detected in the camera frame."""
 
def detect_persons() -> list[dict]:
    """Returns list of detected persons: [{"id": str|None, "confidence": float}]"""
 
def identify_known_person(person_id: str = None) -> bool:
    """Returns True if a known person (optionally specific person_id) is visible."""
 
def count_persons() -> int:
    """Returns the number of persons currently visible."""
 
def get_frame() -> dict:
    """Returns metadata about current frame: {"persons": int, "brightness": float, "timestamp": str}
    Does NOT return raw image data."""
```
 
**`pir.py`**
 
```python
"""
Passive infrared sensor wrapper.
Includes debouncing and hysteresis to filter sensor noise.
"""
 
def motion_detected() -> bool:
    """Returns True if motion was detected within the last debounce window."""
 
def time_since_last_motion() -> float:
    """Returns seconds since last motion detection. Returns -1 if no motion ever detected."""
```
 
**`microphone.py`**
 
```python
"""
Audio input with integrated STT (Whisper).
listen() returns transcribed text, not audio data.
"""
 
def is_speech() -> bool:
    """Returns True if speech is currently detected."""
 
def listen(timeout: int = 10) -> str:
    """Listens for speech, transcribes, and returns text. Returns '' on timeout."""
 
def ambient_level() -> float:
    """Returns ambient noise level as float 0.0-1.0."""
```
 
#### 5.2.2 Output Modules
 
**`speaker.py`**
 
```python
"""
Text-to-speech output.
say_and_wait() is synchronous: speaks, listens for response, returns transcribed text.
This is the core interaction primitive for voice-first skills.
"""
 
def say(text: str, speed: float = 1.0, volume: float = 0.8) -> None:
    """Speaks the text aloud. Non-blocking."""
 
def play_sound(name: str) -> None:
    """Plays a named sound effect (e.g., 'chime', 'alert', 'success')."""
 
def say_and_wait(text: str, timeout: int = 15) -> str:
    """Speaks text, then listens for a response. Returns transcribed response or '' on timeout."""
```
 
**`display.py`**
 
```python
"""
Visual output for devices with a screen.
show_options() presents choices and returns the selection.
"""
 
def show_text(text: str, size: str = "medium") -> None:
    """Shows text on display. size: 'small', 'medium', 'large'."""
 
def show_image(image_ref: str) -> None:
    """Shows a named/referenced image on display."""
 
def show_options(options: list[str]) -> str:
    """Shows selectable options, waits for selection, returns chosen option text."""
 
def clear() -> None:
    """Clears the display."""
```
 
**`light.py`**
 
```python
"""
LED ring / ambient light control.
Non-verbal attention mechanism — gentle pulse instead of beep.
"""
 
def set_color(color: str) -> None:
    """Sets LED to a color. Accepts named colors ('blue', 'green', 'red', 'warm_white')."""
 
def pulse(color: str, pattern: str = "gentle") -> None:
    """Pulses LED. pattern: 'gentle' (slow fade), 'alert' (faster), 'urgent' (rapid)."""
 
def off() -> None:
    """Turns off all LEDs."""
```
 
#### 5.2.3 Communication Modules
 
**`whatsapp.py`**
 
```python
"""
WhatsApp Business API wrapper.
All returned data uses dict format: msg["sender"], msg["text"], etc.
"""
 
def get_unread(contact: str = None) -> list[dict]:
    """Returns unread messages: [{"sender": str, "sender_email": str, "text": str, "timestamp": str}]
    Optional contact filter (fuzzy name match)."""
 
def get_contacts() -> list[dict]:
    """Returns all WhatsApp contacts: [{"name": str, "phone": str, "email": str|None}]"""
 
def send(contact: str, text: str) -> bool:
    """Sends a message to a contact (fuzzy name match). Returns True on success.
    NOTE: This is an effectful action — goes through the Capability Broker."""
 
def mark_read(msg_id: str) -> None:
    """Marks a message as read."""
```
 
**`sms.py`**
 
```python
"""
SMS interface. Identical signature to whatsapp.py for platform-agnostic skill authoring.
"""
 
def get_unread(contact: str = None) -> list[dict]:
    """Returns unread SMS: [{"sender": str, "text": str, "timestamp": str}]"""
 
def send(contact: str, text: str) -> bool:
    """Sends an SMS. Returns True on success. Effectful — goes through Broker."""
```
 
**`email.py`**
 
```python
"""
Email access. summarize_unread() uses an internal LLM call to produce a spoken summary.
"""
 
def get_unread(filter: str = None) -> list[dict]:
    """Returns unread emails: [{"sender_name": str, "sender_email": str, "subject": str, "body": str, "timestamp": str}]
    Optional filter: 'today', 'important', or a sender name."""
 
def send(to: str, subject: str, body: str) -> bool:
    """Sends an email. Effectful — goes through Broker."""
 
def summarize_unread() -> str:
    """Returns a natural-language summary of unread emails (uses internal LLM call with own budget)."""
```
 
**`phone.py`**
 
```python
"""
Telephony control for incoming/outgoing voice calls.
"""
 
def call(contact: str) -> bool:
    """Initiates a call. Effectful — requires Broker approval + explicit user voice confirmation."""
 
def is_ringing() -> bool:
    """Returns True if an incoming call is active."""
 
def get_caller() -> dict:
    """Returns caller info: {"name": str|None, "number": str} or empty dict if no call."""
 
def answer() -> None:
    """Answers an incoming call."""
 
def hangup() -> None:
    """Ends the current call."""
```
 
**`contacts.py`**
 
```python
"""
Address book with fuzzy/semantic search.
Supports relationship tags ("my grandson" → correct contact).
"""
 
def find(name_fuzzy: str) -> dict:
    """Fuzzy-matches a contact name. Returns {"name": str, "phone": str, "email": str, "relationship": str}
    or empty dict if no match. Understands relationship terms ("my daughter", "the doctor")."""
 
def get_all() -> list[dict]:
    """Returns all contacts."""
 
def get_relationship(contact: str) -> str:
    """Returns the relationship tag for a contact: 'grandson', 'doctor', 'neighbor', etc."""
```
 
#### 5.2.4 Reasoning Modules
 
**`llm_call.py`**
 
```python
"""
Pre-packaged LLM operations with built-in token budgets.
No open-ended generation possible. classify() and extract() are preferred
over respond() for cost and determinism.
 
Token budgets per call (hardcoded, not overridable by skills):
  classify:  500 tokens
  extract:   500 tokens
  summarize: 1000 tokens
  respond:   2000 tokens
  sentiment: 200 tokens
"""
 
def classify(text: str, options: list[str]) -> str:
    """Classifies text into one of the given options. Returns the matching option string."""
 
def extract(text: str, schema: dict) -> dict:
    """Extracts structured data from text according to schema.
    Schema format: {"field_name": "type_hint", ...}
    Returns dict matching schema keys."""
 
def summarize(text: str, max_words: int = 50) -> str:
    """Summarizes text in max_words words."""
 
def respond(context: str, instruction: str) -> str:
    """Generates a response given context and instruction. Use sparingly — most expensive call."""
 
def sentiment(text: str) -> str:
    """Returns sentiment: 'positive', 'neutral', 'negative', or 'urgent'."""
```
 
**`memory.py`**
 
```python
"""
Persistent key-value store for skill state.
Intentionally primitive (no SQL, no queries). TTL-based expiry for ephemeral data.
"""
 
def remember(key: str, value: str, ttl: int = None) -> None:
    """Stores a value. Optional TTL in seconds."""
 
def recall(key: str) -> str:
    """Retrieves a value. Returns '' if key doesn't exist or has expired."""
 
def forget(key: str) -> None:
    """Deletes a key."""
 
def recent(key: str, n: int = 10) -> list[str]:
    """Returns the last n values stored under a list-key."""
```
 
**`patterns.py`**
 
```python
"""
Behavioral pattern tracker. Powers Phase 0 trigger detection.
Observes and counts without acting.
"""
 
def track(event_name: str) -> None:
    """Records an occurrence of an event."""
 
def frequency(event_name: str, days: int = 7) -> float:
    """Returns average occurrences per day over the last N days."""
 
def last_time(event_name: str) -> str:
    """Returns ISO timestamp of last occurrence, or '' if never."""
 
def streak(event_name: str) -> int:
    """Returns number of consecutive days the event occurred."""
```
 
#### 5.2.5 Time & Control Modules
 
**`scheduler.py`**
 
```python
"""
Cron-like scheduling with human-readable inputs.
Rate-limited: maximum 20 active schedules per skill.
"""
 
def every(interval: str, callback_name: str) -> str:
    """Schedules recurring execution. interval: '2 hours', '30 minutes', '1 day'.
    Returns job_id."""
 
def at(time: str, callback_name: str) -> str:
    """Schedules execution at a specific time: '08:00', '14:30'. Returns job_id."""
 
def after(delay: str, callback_name: str) -> str:
    """Schedules execution after a delay: '5 minutes', '1 hour'. Returns job_id."""
 
def cancel(job_id: str) -> None:
    """Cancels a scheduled job."""
```
 
**`rules.py`**
 
```python
"""
Event-driven rule engine. Connects sensor conditions to actions.
Cooldown and suppress prevent rapid-fire triggering.
Every rule gets a default cooldown of 60 seconds even if not specified.
"""
 
def when(action_fn, cooldown: str = "60s") -> str:
    """Registers a function to be called on the rule engine's evaluation cycle. Returns rule_id."""
 
def when_all(condition_fns: list, action_fn, cooldown: str = "60s") -> str:
    """Registers an action that fires only when ALL condition functions return True. Returns rule_id."""
 
def cooldown(rule_id: str, interval: str) -> None:
    """Sets minimum interval between rule firings: '5min', '1hour'."""
 
def suppress(rule_id: str, duration: str) -> None:
    """Temporarily disables a rule for the given duration."""
```
 
#### 5.2.6 Environment Modules
 
**`smarthome.py`**
 
```python
"""
Unified smart home interface over Hue, Zigbee, MQTT, etc.
"""
 
def devices(type: str = None) -> list[dict]:
    """Lists connected devices: [{"id": str, "name": str, "type": str, "state": dict}]
    Optional type filter: 'light', 'thermostat', 'plug', 'sensor'."""
 
def turn_on(device: str) -> bool:
    """Turns on a device by name or id. Effectful — goes through Broker."""
 
def turn_off(device: str) -> bool:
    """Turns off a device. Effectful — goes through Broker."""
 
def set(device: str, property: str, value) -> bool:
    """Sets a device property: set('living_room_light', 'brightness', 80). Effectful."""
```
 
**`weather.py`**
 
```python
"""
Weather data. Simple but frequently requested by seniors.
"""
 
def now() -> dict:
    """Returns current weather: {"temp_c": float, "condition": str, "humidity": float}"""
 
def forecast(hours: int = 24) -> list[dict]:
    """Returns hourly forecast for the next N hours."""
 
def will_rain_today() -> bool:
    """Returns True if rain is expected today."""
```
 
**`calendar.py`**
 
```python
"""
Calendar integration for appointments and reminders.
"""
 
def today() -> list[dict]:
    """Returns today's events: [{"title": str, "time": str, "location": str}]"""
 
def tomorrow() -> list[dict]:
    """Returns tomorrow's events."""
 
def next_event() -> dict:
    """Returns the next upcoming event, or empty dict."""
 
def add(title: str, time: str) -> bool:
    """Adds a calendar event. Effectful — goes through Broker."""
 
def remind_before(event_title: str, minutes: int = 30) -> str:
    """Sets a reminder before an event. Returns reminder_id."""
```
 
#### 5.2.7 Safety Module
 
**`safety.py`**
 
```python
"""
Provides read-only context queries and logging.
 
IMPORTANT: Effectful safety checks (user confirmation, rate limiting, escalation to
trusted contacts) are enforced by the Capability Broker in the trusted process,
NOT by the skill itself. The skill cannot bypass policy by omitting a call.
See Section 7.4 for Broker details.
"""
 
def night_mode() -> bool:
    """Returns True if night mode is active (default: 22:00-07:00)."""
 
def log(event: str, severity: str = "info") -> None:
    """Logs an event. severity: 'debug', 'info', 'warning', 'error'."""
 
def disable_skill(reason: str) -> None:
    """Self-disables the current skill with a reason. Notifies the user."""
```
 
### 5.3 MVP Module Set (Launch Target)
 
The following 15 modules are sufficient to cover an estimated 90% of senior daily-use requests:
 
| Category | Modules | Count |
|----------|---------|-------|
| Sensors | camera.py, pir.py, microphone.py | 3 |
| Output | speaker.py, display.py, light.py | 3 |
| Communication | whatsapp.py, contacts.py, phone.py | 3 |
| Reasoning | llm_call.py, memory.py, patterns.py | 3 |
| Time & Control | scheduler.py, rules.py | 2 |
| Safety | safety.py | 1 |
 
Smart home, weather, calendar, SMS, and email modules are Phase 2 additions.
 
---
 
## 6. Example Skills: From Simple to Complex
 
### 6.1 Simple: WhatsApp Read-Aloud
 
A basic skill that reads incoming WhatsApp messages when the user is alone in the room:
 
```python
# skill: whatsapp_read_aloud
# compiled: 2026-03-15T10:23:00Z
 
from modules.camera import is_anyone_visible, count_persons
from modules.whatsapp import get_unread
from modules.speaker import say
from modules.rules import when_all
from modules.safety import night_mode, log
 
def should_trigger():
    return (
        is_anyone_visible()
        and count_persons() == 1
        and not night_mode()
        and len(get_unread()) > 0
    )
 
def read_messages():
    for msg in get_unread():
        say(f'Nachricht von {msg["sender"]}: {msg["text"]}')
        log("whatsapp_read", "info")
 
when_all([should_trigger], read_messages, cooldown="5min")
```
 
15 lines. No boilerplate, no hardware setup, no error handling code — that lives in the modules. The Coding Agent's output is a wiring diagram, not a software project.
 
### 6.2 Complex: Smart Email Router with Cross-Channel Logic
 
A user request like "Check my emails, but only read me the ones from people I also have on WhatsApp. And if someone is asking about my schedule tomorrow, don't read it — just send them my calendar" produces a skill with iteration, cross-module data joins, and LLM-driven intent classification:
 
```python
# skill: smart_email_router
# compiled: 2026-03-15T14:07:00Z
 
from modules.email import get_unread
from modules.whatsapp import get_contacts, send
from modules.calendar import tomorrow
from modules.llm_call import classify, extract
from modules.speaker import say
from modules.contacts import find
from modules.rules import when
from modules.safety import night_mode, log
 
def process_emails():
    if night_mode():
        return
 
    # Cross-reference: emails from people also in WhatsApp
    wa_contacts = get_contacts()
    wa_emails = set()
    for c in wa_contacts:
        if c["email"]:
            wa_emails = wa_emails | {c["email"]}
 
    for mail in get_unread():
        # Only process mails from WhatsApp contacts
        if mail["sender_email"] not in wa_emails:
            continue
 
        # LLM: what does this email want?
        intent = classify(
            mail["body"],
            ["wants_information", "shares_information",
             "action_required", "spam"]
        )
 
        if intent == "wants_information":
            topic = extract(mail["body"], {"topic": "string"})
            if "termin" in topic["topic"]:
                schedule = tomorrow()
                contact = find(mail["sender_name"])
                send(contact, f"Mein Terminplan: {schedule}")
                log("auto_reply_schedule", "info")
            else:
                say(f'Mail von {mail["sender_name"]}: {mail["subject"]}')
 
        elif intent == "shares_information":
            say(f'Mail von {mail["sender_name"]}: {mail["subject"]}')
 
        elif intent == "action_required":
            say(f'Wichtige Mail von {mail["sender_name"]}: '
                f'{mail["subject"]}. Soll ich antworten?')
 
when(process_emails, cooldown="10min")
```
 
This skill is substantially more complex: it joins data across two communication channels, uses LLM classification to route by intent, auto-responds to schedule requests via a different channel, and escalates action-required items. Note the dict-style data access (`mail["sender_name"]` instead of `mail.sender_name`) — this eliminates Python object-model escape vectors while remaining readable.
 
Yet it remains safe: every function call is an RPC request to the trusted parent process, every effect (`send`, `say`) is authorized by the Capability Broker, and the entire script runs inside a sealed process with kernel-level enforcement. The skill cannot bypass policy because it never holds a direct reference to any effectful resource.
 
---
 
## 7. Enforcement Architecture
 
Safety for twinr is not a feature — it's a design constraint. The target user cannot debug, cannot interpret error messages, and will lose trust after a single unexplained failure. At the same time, the system must support genuinely complex skills: cross-channel data joins, LLM-driven classification, conditional multi-step workflows.
 
The design principle is: **restrict access to dangerous resources, not logical complexity.** A skill may contain sophisticated branching, iteration over bounded collections, and multi-module data flows. What it may not do is authorize any effect on the outside world. That authority belongs exclusively to trusted code.
 
> ***The governing axiom: Untrusted Python may compose behavior; only trusted code may authorize effects.***
 
The architecture is organized into three tiers: pre-execution filters (useful defense-in-depth, but explicitly not security boundaries), hard enforcement boundaries (kernel-level, process-level guarantees), and the Capability Broker (the authoritative policy layer).
 
### 7.1 Three Pillars
 
| Pillar | Meaning | Enforced By |
|--------|---------|-------------|
| **Sealed Capability Surface** | The skill process has no access to any system resource except explicitly granted capability proxies. No imports, no builtins, no file descriptors, no syscalls beyond pure computation. | Kernel (seccomp + Landlock), Trusted Loader |
| **Fail-Closed Execution** | Every layer defaults to denial. An unknown AST node is rejected. An unresolved import produces nothing. A missing capability grant blocks the call. A timed-out process is killed. There is no permissive fallback anywhere. | All layers independently |
| **Brokered Effects** | Every action that affects the outside world (send a message, speak aloud, turn on a light, make an LLM call) is authorized by a trusted Policy Broker. The skill cannot bypass this broker because it never holds a direct reference to any effectful resource. | Policy Broker (trusted process) |
 
### 7.2 Tier 1: Pre-Execution Filters (Defense-in-Depth)
 
These filters catch errors and malformed code early. They are valuable for fast feedback and reducing the attack surface, but they are **explicitly not security boundaries**. A determined attacker could theoretically bypass any Python-level filter. The hard enforcement (Tiers 2 and 3) must hold even if Tier 1 fails completely.
 
#### 7.2.1 AST Validator
 
The generated skill script is parsed into a Python AST and walked against a strict whitelist of allowed node types, pinned to the exact Python version deployed on the target hardware (AST grammar changes between releases).
 
**Allowed:** `FunctionDef`, `If`, `For` (bounded iteration over iterables), `BoolOp`, `Compare`, `Return`, `Assign`, `AugAssign`, `Subscript` (dict-style access), `Call`, `Constant`, `Name`, `JoinedStr` (f-strings), `List`/`Dict`/`Set`/`Tuple` literals.
 
**Forbidden (hard reject on detection):** `While`, `Class`, `Lambda`, `Yield`/`YieldFrom`, `Async*`, `Try`/`Except`, `With`, `Match`, `Global`, `Nonlocal`, `Import`/`ImportFrom` (handled by trusted loader instead), `NamedExpr` (`:=` walrus), Decorators, `ListComp`/`SetComp`/`DictComp`/`GeneratorExp` (hidden iteration and scope creation), all `Attribute` access nodes (see 7.2.3).
 
Any single forbidden node → unconditional reject. No exceptions, no fallback.
 
##### Implementation
 
```python
# ast_validator.py — approximately 200 lines
import ast
import sys
 
# Pin to exact Python version
REQUIRED_PYTHON = (3, 11)
assert sys.version_info[:2] == REQUIRED_PYTHON, f"AST validator requires Python {REQUIRED_PYTHON}"
 
ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.AugAssign,
    ast.If, ast.For,  # For is bounded iteration only
    ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
    ast.Call, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.Subscript, ast.Index, ast.Slice,  # dict-style access
    ast.JoinedStr, ast.FormattedValue,  # f-strings
    ast.List, ast.Dict, ast.Set, ast.Tuple,
    ast.Expr, ast.Pass, ast.Break, ast.Continue,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.arguments, ast.arg,
    ast.keyword,
}
 
def validate(source: str) -> tuple[bool, str]:
    """Returns (True, '') if valid, or (False, reason) if rejected."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
 
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return False, f"Forbidden construct: {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
 
    return True, ''
```
 
#### 7.2.2 Import Extraction (No Runtime Imports)
 
The skill process does not have access to Python's import machinery at all. Instead, the **Trusted Loader** (running in the parent process):
 
1. Extracts `import` statements from the AST.
2. Strips them from the source code before execution.
3. Resolves each import against the module allowlist.
4. For each approved import, generates an **RPC proxy stub** (not the real module).
5. Injects the proxy stubs into the skill's execution globals dict.
 
This eliminates the entire attack surface of `sys.meta_path`, `sys.path_hooks`, and `sys.path_importer_cache`. The untrusted process never sees a real Python module object, never gets `__import__`, and has no mechanism to load code dynamically.
 
##### Implementation
 
```python
# trusted_loader.py — runs in PARENT process, never in skill process
import ast
 
MODULE_ALLOWLIST = {
    "modules.camera", "modules.pir", "modules.microphone",
    "modules.speaker", "modules.display", "modules.light",
    "modules.whatsapp", "modules.sms", "modules.email",
    "modules.phone", "modules.contacts",
    "modules.llm_call", "modules.memory", "modules.patterns",
    "modules.scheduler", "modules.rules",
    "modules.smarthome", "modules.weather", "modules.calendar",
    "modules.safety",
}
 
def extract_imports(source: str) -> tuple[str, dict]:
    """Extracts imports from source, returns (clean_source, required_capabilities).
    clean_source has all import statements removed.
    required_capabilities maps name → (module, function) for proxy generation."""
 
    tree = ast.parse(source)
    imports = {}
    lines_to_remove = set()
 
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name not in MODULE_ALLOWLIST:
                raise SecurityError(f"Blocked import: {module_name}")
            for alias in node.names:
                local_name = alias.asname or alias.name
                imports[local_name] = (module_name, alias.name)
            lines_to_remove.add(node.lineno)
 
    # Strip import lines from source
    clean_lines = []
    for i, line in enumerate(source.splitlines(), 1):
        if i not in lines_to_remove:
            clean_lines.append(line)
 
    return "\n".join(clean_lines), imports
 
def build_proxy_stubs(imports: dict) -> dict:
    """Builds RPC proxy stubs for each imported function.
    These stubs serialize the call and send it to the Broker via IPC pipe."""
    globals_dict = {}
    for local_name, (module_name, func_name) in imports.items():
        globals_dict[local_name] = make_rpc_proxy(module_name, func_name)
    return globals_dict
 
def make_rpc_proxy(module_name: str, func_name: str):
    """Returns a callable that serializes args and sends an RPC request
    to the trusted parent process via the IPC pipe."""
    def proxy(*args, **kwargs):
        request = {
            "type": "capability_call",
            "module": module_name,
            "function": func_name,
            "args": args,
            "kwargs": kwargs,
        }
        # Send via IPC pipe to parent → Broker → Module → response
        return ipc_call(request)
    return proxy
```
 
#### 7.2.3 Sealed Data Access (No Attribute Navigation)
 
If a skill receives a module return value like `msg` and can call `msg.sender`, it is one step away from `msg.__class__.__bases__[0].__subclasses__()` — the classic Python sandbox escape chain.
 
Therefore, **skills do not receive Python objects with attribute access.** All data returned from capability proxies is in sealed dict format: `msg["sender"]`, `msg["text"]`, `msg["timestamp"]`. These are plain Python dicts with no `__class__`, `__dict__`, or `__globals__` to traverse.
 
The AST validator rejects all `Attribute` nodes, making dot-notation access syntactically impossible in skill code.
 
This is less ergonomic than `msg.sender`, but it eliminates the entire object-model escape surface. The Coding Agent is instructed to use dict-style access, and the few-shot examples in its prompt reinforce this pattern.
 
#### 7.2.4 Restricted Builtins
 
The skill's execution globals contain a minimal builtins whitelist:
 
**Allowed:** `len`, `range`, `str`, `int`, `float`, `bool`, `list`, `dict`, `set`, `tuple`, `sorted`, `enumerate`, `zip`, `min`, `max`, `any`, `all`, `isinstance`, `print` (redirected to log).
 
**Everything else does not exist in the namespace** — including `eval`, `exec`, `compile`, `type`, `getattr`, `open`, `__import__`.
 
**Important caveat:** Restricted builtins in Python are **explicitly not a security mechanism**. The Python documentation states that `__builtins__` manipulation is not designed for sandboxing, and normal function objects carry `__globals__` and `__builtins__` references. This is why the skill never receives real Python function objects or module objects — only opaque RPC-style capability proxies. The builtins whitelist is a defense-in-depth filter, not a boundary.
 
### 7.3 Tier 2: Hard Enforcement Boundaries
 
These are the **actual security boundaries**. They are enforced by the operating system kernel and are independent of Python entirely. Even a complete CPython escape cannot breach these boundaries.
 
#### 7.3.1 Process Isolation
 
Every skill execution runs in a **separate process** (never a thread). This is non-negotiable: seccomp's `SECCOMP_RET_KILL_THREAD` in a multithreaded process can leave the host in a permanently corrupt state. Process-per-skill provides clean lifecycle management, clean resource accounting, and clean kill semantics.
 
The skill process is spawned with `no_new_privs` set, ensuring it cannot escalate privileges.
 
##### Implementation
 
```python
# skill_runner.py — runs in PARENT process
import os
import json
import subprocess
 
def run_skill(skill_source: str, proxy_stubs: dict, timeout: int = 5):
    """Executes a skill in an isolated subprocess with full enforcement."""
 
    # Create IPC pipe
    parent_read, child_write = os.pipe()
    child_read, parent_write = os.pipe()
 
    pid = os.fork()
    if pid == 0:
        # CHILD PROCESS — this becomes the sandboxed skill process
        os.close(parent_read)
        os.close(parent_write)
 
        # Close ALL other file descriptors
        for fd in range(3, os.sysconf("SC_OPEN_MAX")):
            if fd not in (child_read, child_write):
                try:
                    os.close(fd)
                except OSError:
                    pass
 
        # Apply Landlock (must happen before seccomp)
        apply_landlock_ruleset()  # see 7.3.3
 
        # Apply seccomp filter
        apply_seccomp_filter(child_read, child_write)  # see 7.3.2
 
        # Execute skill with restricted environment
        restricted_globals = build_restricted_globals(proxy_stubs)
        exec(skill_source, restricted_globals)
 
        os._exit(0)
    else:
        # PARENT PROCESS — runs Broker, handles IPC
        os.close(child_read)
        os.close(child_write)
        handle_skill_ipc(pid, parent_read, parent_write, timeout)
```
 
#### 7.3.2 seccomp-bpf (Syscall Filtering)
 
The skill process runs under a seccomp-bpf filter:
 
**Allowed:** `read`/`write` (on inherited IPC pipe FDs only), `mmap`/`mprotect`/`brk` (memory management), `futex` (threading primitives), `clock_gettime`, `exit`/`exit_group`. This is the minimum set for Python to function.
 
**Blocked:** All networking (`socket`, `connect`, `bind`, `listen`, `accept`), all filesystem creation (`open`/`openat` with write flags, `mkdir`, `link`, `unlink`), all process creation (`fork`, `clone`, `execve`), all introspection (`ptrace`, `process_vm_readv`), all filesystem manipulation (`mount`, `umount`, `chroot`).
 
Violation triggers `SECCOMP_RET_KILL_PROCESS` (not `KILL_THREAD`), terminating the entire skill process immediately.
 
#### 7.3.3 Landlock (Filesystem Enforcement)
 
On Linux 5.13+, Landlock LSM provides filesystem-level access control. The skill process is launched with a Landlock ruleset that grants access to **nothing**.
 
Two critical implementation details:
 
1. **File descriptors inherited before sandboxing are not retroactively restricted.** Therefore, the skill process must be spawned with a clean FD table: only the IPC pipe to the trusted parent. stdin/stdout/stderr redirected to /dev/null. All other FDs closed before Landlock ruleset is applied.
 
2. **Landlock features are ABI-version-dependent.** TCP connect/bind enforcement requires ABI v4, IPC scopes require later versions. The runtime must query the available ABI version at startup and adapt the ruleset accordingly, **always failing closed**: if a needed restriction is not available in the running kernel, the capability is denied entirely rather than left unprotected.
 
##### Implementation Notes
 
```python
# landlock.py — applied in child process before seccomp
import ctypes
import os
 
def apply_landlock_ruleset():
    """Apply maximally restrictive Landlock ruleset.
    Must be called AFTER FD cleanup, BEFORE seccomp."""
 
    # Query available ABI version
    abi_version = landlock_create_ruleset(None, 0, LANDLOCK_CREATE_RULESET_VERSION)
 
    # Create ruleset with ALL available handled access types
    # This means: the kernel will enforce restrictions on ALL file operations
    # Since we add NO rules, everything is denied
    ruleset_attr = LandlockRulesetAttr(
        handled_access_fs=ALL_FS_ACCESS_RIGHTS[abi_version],
        handled_access_net=ALL_NET_ACCESS_RIGHTS[abi_version] if abi_version >= 4 else 0,
    )
    ruleset_fd = landlock_create_ruleset(ruleset_attr)
 
    # Add NO rules — deny everything
 
    # Enforce
    prctl(PR_SET_NO_NEW_PRIVS, 1)  # required before landlock_restrict_self
    landlock_restrict_self(ruleset_fd)
    os.close(ruleset_fd)
```
 
### 7.4 Tier 3: The Capability Broker (Brokered Effects)
 
This is the **most important architectural component**. The Capability Broker is the trusted process-level authority that decides whether any effect is permitted.
 
#### 7.4.1 Why the Broker Exists
 
In a naive design, `safety.confirm()` is a function the skill voluntarily calls before performing a sensitive action. This is fundamentally broken: it relies on untrusted code to police itself. A generated skill that omits the `confirm()` call — whether by LLM error or adversarial prompt injection — bypasses the safety check entirely.
 
The Broker inverts this relationship. **The skill never holds a direct reference to any effectful resource.** When a skill calls `send(contact, text)`, it is not calling WhatsApp directly. It is sending an RPC request to the trusted parent process, which routes it through the Broker. The Broker evaluates runtime policy before granting or denying the request.
 
#### 7.4.2 Policy Dimensions
 
The Broker evaluates every effect request against multiple policy dimensions simultaneously:
 
| Dimension | Example Check | Deny Condition |
|-----------|--------------|----------------|
| **Privacy class** | Is this a private message being read aloud? | Others present in camera view |
| **User consent** | Has the user approved this skill's communication scope? | Sending to a contact not in approved scope |
| **Time policy** | Is it night mode? | Non-emergency output between 22:00–07:00 |
| **Rate limits** | How many messages has this skill sent this hour? | Exceeds per-skill hourly cap |
| **Skill risk class** | Is this a newly compiled skill? | New skills get stricter limits for first 72h |
| **Relationship model** | Who is the recipient? | Medical contacts require higher confirmation threshold |
| **Cost budget** | How much has this skill spent on LLM calls? | Exceeds daily token budget |
 
#### 7.4.3 Broker Enforcement Examples
 
- `whatsapp.send()` without a Broker grant → **hard denied**, logged, skill error count incremented.
- `speaker.say()` with content classified as private, while `camera.count_persons() > 1` → **hard denied**, content queued for when user is alone.
- `phone.call()` to any contact → **requires explicit user approval token** (voice confirmation in real time), regardless of what the skill code says.
- `llm_call.respond()` that would exceed the skill's daily token budget → **denied**, skill receives a budget-exceeded error via IPC.
 
The skill code is never aware of these checks. It calls `send()` and either gets a success or a denial. The policy logic lives entirely in the trusted process, unreachable and unmodifiable by the skill.
 
##### Implementation
 
```python
# broker.py — runs in PARENT process
class CapabilityBroker:
    def __init__(self, skill_id: str, skill_manifest: dict):
        self.skill_id = skill_id
        self.manifest = skill_manifest  # from deployment pipeline
        self.call_count = 0
        self.token_spend = 0
        self.send_count_hourly = 0
 
    def evaluate(self, request: dict) -> dict:
        """Evaluates a capability request against all policy dimensions.
        Returns {"granted": True, "result": ...} or {"granted": False, "reason": str}."""
 
        module = request["module"]
        function = request["function"]
        args = request["args"]
        kwargs = request["kwargs"]
 
        # --- Rate limits ---
        self.call_count += 1
        if self.call_count > 50:
            return deny("Capability invocation limit exceeded (50/run)")
 
        # --- Effect classification ---
        effect_class = classify_effect(module, function)
 
        if effect_class == "read_only":
            # Read-only calls (get_unread, is_anyone_visible, etc.) pass through
            return grant(self.execute(module, function, args, kwargs))
 
        if effect_class == "output":
            # Speaking, display — check privacy and night mode
            if self.check_night_mode() and not self.is_emergency(args):
                return deny("Night mode active")
            if self.check_others_present() and self.is_private_content(args):
                return deny("Others present, private content")
            return grant(self.execute(module, function, args, kwargs))
 
        if effect_class == "communicate":
            # Sending messages, making calls — strictest checks
            if not self.check_contact_in_scope(args):
                return deny("Contact not in skill's approved scope")
            self.send_count_hourly += 1
            if self.send_count_hourly > 20:
                return deny("Hourly send limit exceeded")
            if function == "call":
                # Phone calls always require real-time voice confirmation
                if not self.request_user_confirmation(f"Soll ich {args[0]} anrufen?"):
                    return deny("User declined call")
            return grant(self.execute(module, function, args, kwargs))
 
        if effect_class == "llm":
            # LLM calls — check token budget
            estimated_tokens = estimate_tokens(function, args)
            if self.token_spend + estimated_tokens > self.daily_budget:
                return deny("Daily token budget exceeded")
            self.token_spend += estimated_tokens
            return grant(self.execute(module, function, args, kwargs))
 
        return deny(f"Unknown effect class for {module}.{function}")
```
 
#### 7.4.4 Broker as Audit Trail
 
Every Broker decision — grant or deny — is logged with full context: skill ID, capability requested, policy dimensions evaluated, result, timestamp. This audit trail is the basis for the monitoring dashboard (available to the user's trusted contact) and for post-incident analysis.
 
### 7.5 Watchdog (Availability Layer)
 
The watchdog is not a security boundary — it is an **availability and resource protection** mechanism:
 
| Metric | Limit | Action |
|--------|-------|--------|
| Wall-clock time per execution | 5 seconds | SIGKILL process, skill auto-disabled |
| Resident memory (RSS) | 50 MB | SIGKILL process, skill auto-disabled |
| Capability invocations per run | 50 calls | Remaining calls denied via Broker |
| LLM token spend per run | 5,000 tokens | Denied via Broker |
| External messages per hour | 20 sends | Denied via Broker |
| IPC payload size | 1 MB per call | Truncated, logged |
| Error rate (rolling 24h) | >30% fail | Skill auto-paused, user notified |
| User interruption rate | >50% cancelled | Skill flagged for review |
 
The watchdog runs at a higher privilege level and cannot be influenced by skill code.
 
### 7.6 The Deployment Pipeline
 
Complete pipeline from generated code to production:
 
1. **AST Validator** parses and rejects any forbidden construct. Import statements are extracted and stripped.
2. **Trusted Loader** resolves imports against module allowlist. Approved capabilities are compiled into RPC proxy stubs. An **effect manifest** is generated listing every capability the skill will invoke.
3. **Effect manifest check** against system policy: are the requested capabilities configured and available? Are the effect combinations permitted?
4. **Sandbox Test Run:** skill executes in full enforcement environment (restricted builtins, capability proxies, seccomp, Landlock) with mocked Broker responses.
5. **Generated Tests** run in the same environment: happy path, edge cases, and a negative test verifying inertness when trigger conditions are not met.
6. **LLM Review** (optional): a second LLM pass reviews for logical errors and unintended effect patterns.
7. **Guided User Test:** the user walks through a live demonstration. Only after explicit confirmation does the skill enter production.
 
In production, **every skill execution** runs through the full enforcement stack. There is no "trusted mode" for established skills.
 
```
┌─────────────────────────────────────────────────────────────────┐
│                    TRUSTED PARENT PROCESS                       │
│                                                                 │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────────────┐ │
│  │  Module   │  │  Capability   │  │       Watchdog           │ │
│  │  Library  │←─│    Broker     │  │  (resource monitoring)   │ │
│  │ (real     │  │  (policy      │  │                          │ │
│  │  code)    │  │   engine)     │  └──────────────────────────┘ │
│  └──────────┘  └───────┬───────┘                                │
│                        │ IPC pipe (only connection)             │
│  ┌─────────────────────┴───────────────────────────────────────┐│
│  │              SANDBOXED SKILL PROCESS                        ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Skill Script (generated Python)                     │   ││
│  │  │  - Only RPC proxy stubs (no real modules)            │   ││
│  │  │  - Restricted builtins                               │   ││
│  │  │  - Dict-style data access only                       │   ││
│  │  │  - No imports, no attribute access, no while          │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  │  seccomp-bpf: minimal syscalls only                        ││
│  │  Landlock: no filesystem access                             ││
│  │  no_new_privs: no privilege escalation                      ││
│  │  Clean FD table: only IPC pipe inherited                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```
 
### 7.7 Audit & Observability
 
Every enforcement decision is logged: AST rejections, Broker grants and denials, watchdog kills, policy evaluations. The audit log is accessible to the user's trusted contact through a simple dashboard, and provides the basis for post-incident analysis and continuous policy refinement.
 
### 7.8 Escalation & Recovery
 
When the system encounters a situation it cannot resolve, it never fails silently. The escalation chain: inform the user in simple language → offer to contact a trusted person → log the event. If a skill is auto-disabled by the watchdog:
 
> "I've paused [skill name] because it wasn't working correctly. Shall I try to fix it, or would you like to turn it off?"
 
Any skill can be rolled back or disabled with a single voice command: "Forget how to read my emails." The user is always in control.
 
---
 
## 8. Integration with TESSAIRACT / AGENTS.md
 
The Adaptive Skill Engine maps naturally onto the existing TESSAIRACT agent orchestration layer and the AGENTS.md modular skill system:
 
- **Skill-Specs → AGENTS.md Skill Fragments:** Each compiled skill is stored as a modular skill fragment in the AGENTS.md format, with metadata (trigger conditions, required capabilities, compilation timestamp, spec hash) in the header.
- **Coding Agent → Skill Compiler:** The background coding agent functions as a specialized compiler within TESSAIRACT, consuming structured specs and producing executable skill fragments.
- **Module Library → Runtime Primitives:** The module catalog maps to TESSAIRACT's runtime primitive layer, providing the typed, sandboxed interfaces that skill fragments can invoke.
- **Feasibility Check → Capability Registry:** The Phase 1 feasibility check queries the same capability registry that TESSAIRACT uses for agent planning and routing.
 
This means the ASE is not a separate system bolted onto twinr — it's a natural extension of the agent architecture that Arculae is already building.
 
---
 
## 9. Key Engineering Challenges
 
Ordered by estimated difficulty:
 
1. **Senior-friendly requirements dialogue:** The Phase 2 conversation must be precise enough to generate working specs, yet simple enough for a non-technical 78-year-old. Requires extensive prompt engineering and user testing.
2. **Ambiguous answer interpretation:** Extracting structured constraints from natural senior speech ("but not when the grandchildren are here") is a dedicated NLP problem that needs a robust few-shot pipeline.
3. **Validation completeness:** Ensuring generated skills behave correctly in all real-world conditions, not just test scenarios. The guided test (Phase 4) is the last safety net.
4. **Module hardening:** Each module must handle every failure mode gracefully, because the generated skill code above it does not contain error handling. Modules are the only error boundary.
5. **seccomp/Landlock profile tuning:** Configuring kernel-level filters tight enough to protect, loose enough that CPython still works. One-time effort, but requires careful testing.
6. **Capability Broker policy design:** Defining the right policy dimensions, thresholds, and defaults across all effect classes. Must be tunable per-user.
 
---
 
## 10. Success Metrics
 
- **Skill compilation success rate:** >85% of user requests result in a working skill on first attempt
- **Time to learn:** median <30 minutes from request to deployed skill
- **Skill retention:** >80% of learned skills remain active after 30 days
- **Zero-debug rate:** >95% of issues resolved without involving a technical support person
 
---
 
## 11. Why This Changes the Game
 
No existing consumer assistant has this capability. Alexa, Google Home, and Siri all require developers to build skills/actions and users to install them. Apple's Shortcuts require technical proficiency. None of them learn from user behavior and compile new functionality at runtime.
 
The closest analogy in research is NVIDIA's Voyager (an agent that writes its own skills in Minecraft), but Voyager operates in a virtual sandbox with no safety constraints and no non-technical end users.
 
twinr's Adaptive Skill Engine combines the self-improvement paradigm of Voyager with the safety requirements of a medical-grade assistive device and the UX constraints of a product for the least technical user demographic. That intersection is unoccupied.
 
The moat is threefold: the Module Library (curated, tested, domain-specific primitives), the senior-optimized requirements dialogue (tuned through real user testing), and the integration with TESSAIRACT's agent orchestration layer (which provides the compilation and runtime infrastructure). Each of these compounds over time as more skills are compiled and more user patterns are observed.
 
