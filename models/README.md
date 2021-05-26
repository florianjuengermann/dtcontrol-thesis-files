## Generating the Cruise Controllers
You can generate the controller dump with the following steps:
1. download and install UPPAAL Stratego [from here](https://people.cs.aau.dk/~marius/stratego/download.html)
2. run the model checker with `./bin-Linux/verifyta --print-strategies <outputFolder> cruise_250.xml`
3. rename the file `<outputFolder>/safe` to `cruise_250.dump`
4. (optional) create a config file with name `cruise_250_config.json` in the same directory

or just use the generated controllers (in folder `controllers_cps`) after unziping them (`unzip cruise_250.dump.zip`).