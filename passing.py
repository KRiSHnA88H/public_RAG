import embedding as eb
import dataset as ds

for i, chunk in enumerate(ds.dataset):
  eb.add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(ds.dataset)} to the database')


