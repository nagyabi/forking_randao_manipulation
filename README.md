Repository for partially evaluating ETH forking RANDAO manipulations.

1. Collecting beaconchain information from the internet (https://etherscan.io/, https://beaconscan.com/, https://beaconcha.in/)
2. Processing collected data looking for potential anomalies (RANDAO manipulations).
3. Evaluating effects forking attacks (theoretical calculations) with applied heuristics. Providing model which outputs ``best`` actions for the adversary (simulating and testing purposes). We do not advise you on using this model for financial gains.

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
1. Visit [beaconscan.com](https://beaconscan.com/)
2. In a new tab open Chrome console (F12)
3. Visit [beaconscan.com/validators](https://beaconscan.com/validators)
4. Copy the header of the request not including keys starting with ``:``
5. Make beaconscan.header in this [folder](./data/internet/headers/), and paste the header.
6. Test the header with ``python3 -m main --data test-scrape``

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
python3 -m main --theory --size-prefix <PREFIX_SIZE> --size-postfix <POSTFIX_SIZE> --alphas <ALPHAS> [--markov-chain] [--quant]
```
where you can give <ALPHAS> as a single number or a sequence like ``0.01:0.3:0.02`` (start:stop:step).