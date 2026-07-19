import os
import pandas as pd

mapping = {
    "moi_jao": "Moi jao",
    "moi_aho": "Moi aho",
    "moi_khao": "Moi khao",
    "moi_porho": "Moi porho",
    "moi_khelo": "Moi khelo",
    "moi_gaan_gao": "Moi gan gao",
    "moi_bidyaloy_jao": "Moi bidyaloy jao",
    "moi_asomot_thako": "Moi Asomot thako",
    "asomot_bohu_saah_bagisa_ase": "Asomot bohu sah bagisa ase",
    "bihu_asomor_jatiyo_uthob": "Bihu Asomor Jatiyo Uthob"
}

rows = []

for file in os.listdir("wav_recordings"):
    if file.endswith(".wav"):

        for prefix, text in mapping.items():

            if file.startswith(prefix):

                rows.append([
                    os.path.join("wav_recordings", file),
                    text
                ])

                break

df = pd.DataFrame(rows, columns=["audio", "text"])

df.to_csv("dataset.csv", index=False)

print("dataset.csv created!")
print("Total samples:", len(df))