import compiam
from collections import defaultdict
import random
import requests
import os

token = "accc3ed61b727e9c09e3ec70ed1e8867542a4253"
carnatic_corpora = compiam.load_corpora("carnatic", token=token)
recordings = carnatic_corpora.get_collection(recording_detail=True)
data_dir = "top30raga_data"

print("Successfully loaded carnatic corpora.")


"""
TODO
We want 50 clips per raga, each 5 seconds long. We want 10 tracks per raga with 5 clips each. Thus we would like each track to have a length of
    between 250-500 seconds.
1. For each of the top 30 ragas, load all available tracks. Start a counter of 10.
2. Iterate through the available tracks in random order, and check current track's length. If length is between 250-500 seconds, attempt to download.
3. If success, decrement the counter by 1. If counter is 0, break.
4. If not success, continue to the next track.
5. If all tracks are checked, and counter is not 0, repeat the process for the next raga.
6. Continue until all raga directories are populated with 10 tracks.
"""

top_30_ragas = ['Rāgamālika', 'Tōḍi', 'Bhairavi', 'Kamās', 'Kalyāṇi', 'Saurāṣtraṁ',
                 'Śankarābharaṇaṁ', 'Kāmavardani/Pantuvarāḷi', 'Behāg', 'Bēgaḍa',
                   'Sindhubhairavi', 'Mōhanaṁ', 'Sāvēri', 'Kāṁbhōji', 'Rītigauḷa',
                     'Kāpi', 'Nāṭa', 'Suraṭi', 'Pūrvīkaḷyāṇi', 'Hamsadhvani',
                       'Madhyamāvati', 'Hindōḷaṁ', 'Karaharapriya', 'Ānandabhairavi',
                        'Mukhāri', 'Kānaḍa', 'Ābhōgi', 'Nāṭakurinji', 'Śrī', 'Sencuruṭṭi']

# mappings for raga to directory name
raga_to_dir = {
    'Rāgamālika': 'ragamalika',
    'Tōḍi': 'todi',
    'Bhairavi': 'bhairavi',
    'Kamās': 'kamas',
    'Kalyāṇi': 'kalyani',
    'Saurāṣtraṁ': 'saurastram',
    'Śankarābharaṇaṁ': 'sankarabharanam',
    'Kāmavardani/Pantuvarāḷi': 'kamavardanipantuvarali',
    'Behāg': 'behag',
    'Bēgaḍa': 'begada',
    'Sindhubhairavi': 'sindhubhairavi',
    'Mōhanaṁ': 'mohanam',
    'Sāvēri': 'saveri',
    'Kāṁbhōji': 'kambhoji',
    'Rītigauḷa': 'ritigaula',
    'Kāpi': 'kapi',
    'Nāṭa': 'nata',
    'Suraṭi': 'surati',
    'Pūrvīkaḷyāṇi': 'purvikalyani',
    'Hamsadhvani': 'hamsadhvani',
    'Madhyamāvati': 'madhyamavati',
    'Hindōḷaṁ': 'hindolam',
    'Karaharapriya': 'karaharapriya',
    'Ānandabhairavi': 'anandabhairavi',
    'Mukhāri': 'mukhari',
    'Kānaḍa': 'kanada',
    'Ābhōgi': 'abhogi',
    'Nāṭakurinji': 'natakurinji',
    'Śrī': 'sri',
    'Sencuruṭṭi': 'sencurutti'
}
dir_to_raga = {value: key for key, value in raga_to_dir.items()}

# collect recordings for top 30 ragas
top_30_recs = defaultdict(list)
for recording in recordings:
    raaga = recording['raaga'][0]['name'] if len(recording['raaga']) > 0 else None
    if raaga is not None:
        if raaga in top_30_ragas:
            top_30_recs[raaga].append((recording['mbid'], recording['title'], recording['length']))

print("Successfully collected recordings for top 30 ragas.")

# download tracks for top 30 ragas
for raga in top_30_ragas:
    print(f"Downloading tracks for {raga}...")
    recs = top_30_recs[raga]
    random.shuffle(recs)
    count = 10
    dir_path = os.path.join(data_dir, raga_to_dir[raga])
    for rec in recs:
        mbid, title, length = rec
        if length > 250000 and length < 500000:
            try:
                print(f"  Downloading: {title} ({mbid})")
                carnatic_corpora.download_mp3(mbid, dir_path)
                count -= 1
                print(f"  Success!")
            except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError) as e:
                print(f"  Failed to download {title}: {e}")
                continue  # Skip to the next track
            except Exception as e:
                print(f"  Unexpected error for {title}: {e}")
                continue  # Skip to the next track
        if count == 0:
            break


