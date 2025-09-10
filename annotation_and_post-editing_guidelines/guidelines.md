# Annotation and Post-Editing Guidelines

We provide here the annotation guidelines used in our project.

Annotators were tasked with **annotating** and **post-editing** machine-translated passages from English into German, with a focus on studying diverse ambiguity scenarios for seed nouns and their gender representation.

### Annotation of English-language Passages
Each passage consists of four sentences: two as preceding context, one containing the seed word (matching sentence), and one as trailing context. The seed word appears in plural form (e.g., _dispatchers_). Annotators were instructed to:

* Annotate all human entities in the matching sentence, including the seed noun
* Indicate whether the gender of the seed word is **ambiguous**
* If unambiguous, annotate spans that disambiguate the gender in any of the sentences (preceding, matching, or trailing)
* Specify the seed's gender as either **feminine**, **masculine**, **non-binary**, **all genders**, or **feminine and masculine**

For example, in the sentence "In theatrical performances the use of decorative electricity was used in costumes of **female performers**," the gender of the seed word 'performers' is disambiguated by the adjective `female`.

Only the seed noun should be annotated (e.g., in "International Commission of Jurists" only "jurists" should be annotated as 'entity'). Nominal phrases containing the seed should be annotated as 'compounds'.

### Annotation and Post-editing of German Translations

For German translations, annotators were asked to perform the following:

### Annotation:
* Indicate if the gender of the seed is ambiguous in the translation
* If unambiguous, classify it as `masculine`, `feminine`, `non-binary`, `all genders`, or `feminine and masculine`
* Determine whether the gender of the translated seed matches the gender in the source passage

### Post-editing:
1. Fix major errors related to grammar, orthography, and semantics (not stylistic issues)
2. If the gender of the seed was correctly translated, proceed to the next passage
3. If the gender was incorrectly translated, provide appropriate alternatives:
* For masculine source: provide German translation in masculine form
* For feminine source: provide German translation in feminine form
* For non-binary/all genders/feminine and masculine/ambiguous sources, provide **three gender-fair alternatives**:
           1. Gender-neutral rewording
           2. Gender star (*) format
           3. Ens-forms
