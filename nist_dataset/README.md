Data obtained from: [NIST Ultrahigh Carbon Steel Micrographs dataset](https://materialsdata.nist.gov/handle/11256/940), Creative Commons Licenses

Credits to original authors:
- Hecht, Matthew D.
- DeCost, Brian L.
- Francis, Toby
- Holm, Elizabeth A.
- Picard, Yoosuf N.
- Webler, Bryan A.

`.tif` or `.png` format image dataset. Information stored in `microstructures.sqlite` file:
# Windows:
- Get sqlite3 executable (.exe)
- Open cmd, cd to folder with `.sqlite` file
- `.\sqlite3 .\microstructures.sqlite`
- `SELECT * FROM micrograph;`
- Use `pragma table_info(micrograph)` to obtain information on table
