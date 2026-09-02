# Writing Standard for Lesson Text

Lesson READMEs are teaching text. They should read like a chapter of a good textbook or a carefully edited set of lecture notes: complete sentences, ideas introduced before they are used, and every claim accompanied by its reason. They should not read like a slide deck, a blog post, or a chat reply. Every lesson is edited against the rules below, and `TEMPLATE.md` defers to this file for all questions of prose.

## Register

1. **Write in paragraphs of complete sentences.** A paragraph introduces its topic, explains the mechanism, and then states the consequence, in that order. A reader meeting the topic for the first time should be able to follow the paragraph from its first sentence without already knowing the conclusion.
2. **Give the reason with the claim.** Do not leave a claim standing alone as a punchline. Say what the reader will observe, why it happens, and what it implies, in ordinary sentences. For example, do not write "The tree is the argument for metadata-resolved boundaries." Write "The directory tree shows fifty episodes stored in only a handful of parquet and video files, which means the episode boundaries must be recorded somewhere other than the filesystem."
3. **Do not use bold phrases as sentence substitutes.** A bold run-in such as **The three pillars.** is acceptable only when it names a defined concept, and the sentence that follows must be complete and self-contained without it. Prefer a short subheading (`###`) over a bold fragment when a Principles section has several distinct ideas.
4. **No slogans, aphorisms, or one-line verdicts.** Lines such as "Boring is the goal" or "A surprise without a follow-up is an anecdote" are not explanations. If the idea matters, write it as a full sentence with its justification.
5. **Do not use a dash to bolt a comment onto a sentence.** Constructions like "— that is the lesson" or "— the classic bug" are asides that pretend to be arguments. Write a second sentence that says what you mean.
6. **Vary sentence length and use subordinate clauses where they carry the logic.** Words like "because", "so that", "which means", and "whereas" are how reasoning is expressed in prose. A run of short declarative sentences reads as assertion rather than explanation.
7. **Define a term before you use it, and then use the same term.** Do not alternate between synonyms for variety.
8. **Address the reader as "you" in instructions.** Avoid hype, self-reference, and editorializing about the material ("the fun part", "the lesson's signature figure", "the point of the lesson", "the classic"). Let the explanation carry the weight.
9. **Exercise titles are descriptive, not clever.** "Inspect the on-disk layout" rather than "Walk the bytes".

## Structure within a lesson

10. **The opening paragraph** says, in two to four sentences, what the lesson is about, what you will do, and why it matters for the lessons that follow.
11. **Each Principles paragraph handles one idea**, and consecutive paragraphs are connected: say how the next idea follows from the previous one. Use `###` subheadings when the section covers several distinct ideas.
12. **Each exercise opens with a short paragraph** of two to four sentences that says what you will do, which principle it exercises, and what result you are looking for. Numbered steps follow the paragraph.
13. **A [Predict → Run] exercise** says in a full sentence what to write down before running and why the prediction is worth making, and gives the reader the terms in which to predict: a number, an ordering, a curve shape, a sign.
14. **Carry-forward bullets are complete sentences** that state the principle and its reason. Each should make sense to someone who has not read the lesson.
15. **Checkpoints state an observable outcome** and, where it helps, what a deviation from it would indicate.

## What stays terse

16. Tables (the header table, the reading table, hyperparameter tables, pitfalls, deliverables) may use fragments in their cells.
17. Commands, code blocks, and equations are verbatim and are not changed by this standard.
18. Self-check questions are single questions and need no preamble.
19. The style preferences that apply to conversational replies do not apply to files in this repository; this document does.
