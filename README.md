### Datasets : 
- sonnet : textes fixes, utile pour tester le prefix caching.
- random : prompts aléatoires de longueur contrôlée. Idéal pour isoler un effet (« que se passe-t-il avec des prompts de 2000 tokens ? »).
- sharegpt : vraies conversations, longueurs réalistes et variées. C’est la référence la plus utilisée. Il faut télécharger le fichier une fois :
  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

  