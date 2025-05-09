This is the official repository for the paper [Forking the RANDAO: Manipulating Ethereum’s Distributed Randomness Beacon](https://eprint.iacr.org/2025/037.pdf). It includes theoretical evaluations of forking RANDAO manipulations and tools for identifying anomalies on Ethereum mainnet.

1. Collecting beaconchain information from the internet (https://etherscan.io/, https://beaconscan.com/, https://beaconcha.in/)
2. Processing collected data looking for potential anomalies (RANDAO manipulations).
3. Evaluating effects forking attacks (theoretical calculations) with applied heuristics. Providing model which outputs ``best`` actions for the adversary (simulating and testing purposes). We do not advise you on using this model for financial gains.

🚨 Already collected data available until 2025.05.02! If you're only interested in analyzing potential manipulation events, you can skip the data collection phase entirely. Download the full datasets in CSV format from [Google Drive](https://drive.google.com/drive/folders/1uuYVHHhBIOjuCm3qIGdx4rmWUBbL8FeP?usp=sharing), pre-processed and ready to explore. Further release notes in the shared folder.

Note that these CSV files are not compatible with the internal data processing tools in this repo, which use JSON and pickle formats. They're designed to be lightweight and easy to parse so you can analyze the data however you like.

If the data or tools in this repo help your work, consider starring the project ⭐ to support future research.

Good hunting. 🎯

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

For your entity mapping, insert to:
Paste it into [entities.json](./data/jsons/entities.json)
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

Verify BLS signatures (RANDAO reveals) with
```bash
python -m verification.bls_verify
```

**Data Delivery**

When delivering data, we applied cutting heuristics and only regard a certain region of epoch.
size_prefix and size_postfix parameters sets how much information we kepp during a delivery (recommended 2, 8 respectively). The ``alternatives computation`` is time consuming, but can be stopped with C-c and continued later.
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

Try the model in different attacking scenarios described by an extended attack string. First, a quantized model is needed, run the above command with the ``--quant`` flag, then run:
```bash
python3 -m main --theory --size-prefix <PREFIX_SIZE> --size-postfix <POSTFIX_SIZE> --iterations <ITERATIONS> --alphas <ALPHA> --try-quantized
```

To use the model in a different environment, implement a custom agent, that inherits from ``RANDAODataProvider`` (See [this](./theory/method/quant/base.py) file).

**Testing**

```bash
python3 -m pytest tests/
```