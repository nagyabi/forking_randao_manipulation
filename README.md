This is the official repository for the paper [Forking the RANDAO: Manipulating Ethereum’s Distributed Randomness Beacon](https://eprint.iacr.org/2025/037.pdf). It includes theoretical evaluations of forking RANDAO manipulations and tools for identifying anomalies on Ethereum mainnet.

1. Collecting beaconchain information from the internet (https://etherscan.io/, https://beaconscan.com/, https://beaconcha.in/)
2. Processing collected data looking for potential anomalies (RANDAO manipulations).
   - Formatting data to human readable deliveries.
   - Reproducing plots and tables from the article with fresh data.
3. Evaluating effects forking attacks (theoretical calculations) with applied heuristics. Providing model which outputs ``best`` actions for the adversary (simulating and testing purposes). We do not advise you on using this model for financial gains.

🚨 Already collected data available until 2025.05.02! You can skip the data collection (and processing) phase entirely. Download the full datasets in CSV format for independent processing, or in JSON/PICKLE format from [Google Drive](https://drive.google.com/drive/folders/1uuYVHHhBIOjuCm3qIGdx4rmWUBbL8FeP?usp=sharing). Release notes are in the `csv` folder. Download the folders `jsons` and `processed_data` into `data/jsons` and `data/processed_data` respectively.

If the data or tools in this repo help your work, consider starring the project ⭐ to support future research.

Good hunting

**Environment**

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:

   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**Collecting data**

Get API keys here:
* https://beaconcha.in/pricing
* https://etherscan.io/apis#pricing-plan

Add them with the following command:
```bash
python3 -m main --config-api-keys beaconcha --action append --key-values <YourAPIKey> --test-values
python3 -m main --config-api-keys etherscan --action append --key-values <YourAPIKey> --test-values
```

For scraping beaconscan.com:
1. Visit [beaconscan.com/validators](https://beaconscan.com/validators)
2. Open Chrome console (F12 or Right-Click and Inspect)
3. Click on the sorting arrow next to `INDEX`
4. Navigate to the Network tab, and choose the request starting with `datasource?q=validators`
5. The url we are looking for must start with `https://beaconscan.com/datasource?q=validators&type=total&networkId=&sid=<YourSessionID>&draw=`. Copy your session id to a new file named `sid.txt` in this [folder](./data/internet/headers/)
6. Copy the Response Headers starting from `Accept:`
7. Make `beaconscan.header` in to the same [folder](./data/internet/headers/), and paste the header.
8. Test the header with ``python3 -m main --data test-scrape``

There is no guarantee scraping wont broke eventually. If it used to work for you but your header and session id is old, you can try repeating the above process.

For your entity mapping, insert to (or use ours from [Google Drive](https://drive.google.com/drive/folders/1uuYVHHhBIOjuCm3qIGdx4rmWUBbL8FeP?usp=sharing)): [entities.json](./data/jsons/entities.json)
in the format of address: entity name
```json
{
   "0xfddf38947afb03c621c71b06c9c70bce73f12999": "Lido",
   ...
}
```

Start grabbing data with:
```bash
python3 -m main --data --full
```

(Optional) Verify BLS signatures (RANDAO reveals) with
```bash
python -m verification.bls_verify
```

(Optional) For MEV correlations
```bash
python3 -m data.process.collect_reorgs
python3 -m data.process.mev_grinder
```

**Data Delivery**

When delivering data, we applied cutting heuristics and only regard a certain region of epoch.
size_prefix and size_postfix parameters sets how much information we keep during a delivery (recommended 2, 8 respectively). The ``alternatives computation`` is time consuming, but can be stopped with C-c and continued later.
```bash
python3 -m main --alternatives --size-postfix <PREFIX_SIZE> --size-prefix <POSTFIX_SIZE>
python3 -m main --statistics --size-postfix <PREFIX_SIZE> --size-prefix <POSTFIX_SIZE> --export-folder <DELIVERY_PATH>
```

**Theoretical calculations**

For computing the theoretical results, use:
```bash
python3 -m main --theory --size-prefix <PREFIX_SIZE> --size-postfix <POSTFIX_SIZE> --iterations <ITERATIONS> --alphas <ALPHAS> [--markov-chain] [--quant]
```
where you can give <ALPHAS> as a single float between 0 and 1 or a sequence like ``0.01:0.3:0.02`` (start:stop:step).
Similarly, for baselining the plots later with the `only selfish mixing` strategy, use:
```bash
python3 -m main --theory --selfish-mixing --alphas <ALPHAS>
```

Try the model in different attacking scenarios described by an extended attack string. First, a quantized model is needed, run the above command with the ``--quant`` flag, then run:
```bash
python3 -m main --theory --size-prefix <PREFIX_SIZE> --size-postfix <POSTFIX_SIZE> --iterations <ITERATIONS> --alphas <ALPHA> --try-quantized
```

To use the model in a different environment, implement a custom agent, that inherits from ``RANDAODataProvider`` (See [this](./theory/method/quant/base.py) file).

**Reproducing plots and tables**

For this section you will either need processed data (delivery) or theoretical calculations.
Note that in the article we used tikz, but the library the repository is using is not working in every environment.

Figure 6:
`cache` folder stores the utility functions like this (`cache/alpha=0_281-size_prefix=2-size_postfix=6/expected_values.json`)
```json
{"10": {"RO": 9.744380362449885, ...}}
```

Figure 7:
```bash
python3 -m make_figures.theory.expected_values
```
Figure 8:
```bash
python3 -m make_figures.theory.probabilities
```
Table 2:
cache

Figure 9:
```bash
python3 -m make_figures.data.missed_heatmap
```
Table 3:
```bash
python3 -m make_figures.data.bnw_by_entities
```
Figure 10:
```bash
python3 -m make_figures.data.proposed_by_entities
```
Table 4:
```bash
python3 -m data.hypothesis.proposed
```
Table 5:
```bash
python3 -m data.hypothesis.missed
```
Figure 11:
```bash
python3 -m make_figures.theory.one_shot
```
Figure 12:
```bash
python3 -m make_figures.theory.quality_of_eth
```
Figure 13:
```bash
python3 -m make_figures.data.mev_and_normalized_RO
```


**Testing**

```bash
python3 -m pytest tests/
```