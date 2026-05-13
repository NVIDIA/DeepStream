Security support in the Kafka adaptor for DeepStream supports two authentication mechanisms:
2-way TLS and SASL/Plain. Both these mechanisms also support data encryption using TLS.
This document provides detailed instructions to setup and use these mechanisms. 

# Broker Setup 
Both authentication mechanisms involve common broker setup steps relating to deployment of
certificates. These are outlined in this subsection.

Instructions are provided based on [Confluent's](https://www.confluent.io/) Kafka offering;
but the high level procedure applies to other distributions of Kafka as well. Note that TLS
has supplanted SSL, through the terms are used interchangeably in literature.

## Install Kafka broker (skip if you already have a broker)
Install Kafka broker using confluent Kafka distribution based on [Quickstart local](https://docs.confluent.io/current/quickstart/ce-quickstart.html#ce-quickstart) recipe.

Create topic and verify that deepstream Kafka client (eg: test app for Kafka) is able to
connect to your broker setup without security enabled.

## Setup broker security (skip if signed broker certificate is already available)
These steps (a) setup a CA, (b) create certificate for the broker (c) signs the certificate
with the CA. Follow the steps as required for your setup depending on whether you have a CA
and/or signed broker certificate already setup.

### Create CA 
Note that this section creates your own CA for broker certificate signing. In a real world
deployment, it is likely that an established third party CA will be used based on
[certificate signing request(CSR)](https://en.wikipedia.org/wiki/Certificate_signing_request).

Example:  
```
openssl req -new -x509 -keyout ca-key -out ca-cert -days 500
```

Note:  
- ca-key is the file where the generated private key that is associated with the certificate
is stored (user is prompted for password to protect this)  
- ca-cert is certificate
 
### Create certificate for broker and add to keystore 
Example:
```
keytool -keystore kafka.server1.keystore.jks -alias brokerkey -genkey
```
Note:  
- While entering information requested upon running this command, ensure that CN matches the
fully qualified domain name (FQDN) of the server.  
- This command creates both a key and keystore; and adds key to keystore  
- Kafka.server1.keystore.jks is the keystore file  
- brokerkey is the alias name of the key that is generated and added to the keystore  

### Export the certificate from the keystore:
```
keytool -keystore kafka.server1.keystore.jks -alias brokerkey -certreq -file cert-file-server1
```
 
### Sign broker certificate using CA
```
openssl x509 -req -CA ca-cert -CAkey ca-key -in cert-file-server1 -out cert-signed-server1 -days 500 -CAcreateserial
```
Note:  
Use password for ca key provided when generating the CA

### Import CA cert into keystore & truststore
```
keytool -keystore kafka.server1.keystore.jks  -alias CARoot -import -file ca-cert
keytool -keystore kafka.server1.truststore.jks  -alias CARoot -import -file ca-cert
 ```

### Import signed broker cert into keystore
keytool -keystore kafka.server1.keystore.jks  -alias brokerkey -import -file cert-signed-server1


# 2-way TLS authentication
[2-way TLS authentication](https://docs.oracle.com/cd/E19424-01/820-4811/aakhe/index.html)
enables client authentication and data encryption based on TLS/SSL certificates. This section
provides instructions for setting up SSL security for Kafka communication, as supported in
DeepStream. Make sure to first follow instructions in the **Broker Setup** section before
proceeding with these steps. 


## Setup deepstream client security
### Create client CA (skip if desired CA already exists)  
Note that this section creates a CA for the deepstream client. In a real world deployment, it
is likely that an established third party CA will be used. In such a case a certificate
signing process (CSR) will be used.

Example:
```
openssl req -new -x509 -keyout ca-client-key -out ca-client-cert -days 500
```

### Create certificate for deepstream client and add to client keystore 
```
keytool -keystore kafka.client1.keystore.jks -alias dskey -genkey
```

### Export the certificate from the client keystore:
```
keytool -keystore kafka.client1.keystore.jks -alias dskey -certreq -file cert-file-client1
 ```
 
### Sign DS application certificate using client CA (either the CA you created in earlier
step, or one you already had)
```
openssl x509 -req -CA ca-client-cert -CAkey ca-client-key -in cert-file-client1 -out cert-signed-client1 -days 500 -CAcreateserial
```
 
Import client CA cert into client keystore
```
keytool -keystore kafka.client1.keystore.jks  -alias CARoot -import -file ca-client-cert
```

Import signed broker cert into keystore
```
keytool -keystore kafka.client1.keystore.jks  -alias dskey -import -file cert-signed-client1
```

### Convert jks keys to pkcs format on client
This step is required since deepstream (librdkafka) only supports [pkcs#12](https://en.wikipedia.org/wiki/PKCS_12) key format


Step 1 : Create p12 format of keystore  
```
keytool -importkeystore -srckeystore ./kafka.client1.keystore.jks -destkeystore ./kafka.client1.keystore.p12 -deststoretype pkcs12
```

Step 2 : Export the Private Key as a PEM file
```
openssl pkcs12 -in kafka.client1.keystore.p12  -nodes -nocerts -out client1_private_key.pem
```

Step 3 : Exporting the Certificate
```
openssl pkcs12 -in kafka.client1.keystore.p12 -nokeys -out client1_cert.pem
```

### Ensure librdkafka has been built with SSL
Kafka support in DeepStream is built using the librdkafka library. Ensure that librdkafka has
been built with SSL support as described in the README.


### Copy client CA to the broker trust store
Copy client CA certificate (ca-client-cert) to the broker.
Add certificate to the broker truststore:
```
keytool -keystore kafka.server1.truststore.jks  -alias CAClientRoot -import -file ca-client-cert
```

## Configure broker settings to use SSL
Configure Kafka broker to use SSL based on instructions in the
[Configure Brokers](https://docs.confluent.io/current/security/security_tutorial.html#configure-brokers)
section of this [Confluent Quickstart security tutorial](https://docs.confluent.io/current/security/security_tutorial.html).

The main configuration file for Kafka is server.properties located within <install path>/etc/kafka
The tutorial describes changes to be made to this file to enable SSL.

The relevant modified contents of server.properties is shown below. The user should modify fields
such as paths and password as appropriate while using this as reference.
```
# Enable SSL security protocol  
listeners=SSL://:9093 
security.inter.broker.protocol=SSL
ssl.client.auth=required

# Broker security settings
ssl.truststore.location=/var/ssl/private/kafka.server1.truststore.jks
ssl.truststore.password=test1234
ssl.keystore.location=/var/ssl/private/kafka.server1.keystore.jks
ssl.keystore.password=test1234
ssl.key.password=test1234
 

# ACLs
authorizer.class.name=kafka.security.auth.SimpleAclAuthorizer
super.users=User:kafkabroker
allow.everyone.if.no.acl.found=true
```

Note: 
- While the tutorial enables secure communication between Kafka and zookeeper as well (using
SASL), instructions in this section do not enable this functionality and doing so is left as
an option to the user.  
- For sake of simplicity, the example enables authorization for authenticated users to access
Kafka broker topics if no relevant ACL is found - based on the *allow.everyone.if.no.acl.found*
entry. User should modify this to define ACL rules to suit their needs before deploying their
broker. Refer to [Authorization for kafka broker](https://docs.confluent.io/current/kafka/authorization.html)
documentation for details.

## DeepStream Application Changes and Configuration
The Kafka test application needs to be modified to use as part of the connection operation
the correct broker address and port used by the broker for SSL based connections as
configured in server.properties file. Kafka support in DeepStream is built around the
[librdkafka library](https://github.com/edenhill/librdkafka). The kafka configuration options
provided by the Kafka message adaptor to the librdkafka library needs to be modified for SSL.
The DeepStream documentation describes various mechanisms to provide these config options,
but this document addresses these steps based on using a dedicated config file name
config.txt. Note that the config file used by the Kafka test application is defined at the
top of the file as part of the CFG_FILE macro.

A few parameters need to be defined as using the proto-cfg entry within the message-broker
section in this config file as shown in the example below.
```
[message-broker]
proto-cfg = "security.protocol=ssl;ssl.ca.location=<path to your ca>/ca-client-cert;ssl.certificate.location=<path to your certificate >/client1_cert.pem;ssl.key.location=<path to your private key>/client1_private_key.pem;ssl.key.password=test1234;debug=broker,security"
```
The various options specified in the config file are described below: 
```security.protocol=ssl```: use SSL as the authentication protocol  
```ssl.ca.location```: path where your client CA certificate is stored  
```ssl.certificate.location```: path where your client certificate is stored  
```ssl.key.location```: path where your protected private key is stored  
```ssl.key.password```: password for your private key provided while extracting it from the p12 file  
```debug```: enable debug logs for selected categories

Additional SSL based options for rdkafka can be provided here such as the cipher suite used
(ssl.cipher.suites) and also non-security related options such as debug. Refer to the
[librdkafka configuration page](https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md)
for a complete list of options.

### Run the application 
Run the application as before:
```
./test_kafka_proto_sync
```

If all went well, you should see logs relating to connection being established, SSL handshake
and ultimately the messages being sent.

## Viewing messages
You can run a Kafka consumer to receive messages being sent. 
Confluent Kafka is distributed with a Kafka consumer named kafka-console-consumer that you
can use for testing based on the command line below:
```
bin/kafka-console-consumer --bootstrap-server <address>:9093 --topic <your topic> --consumer.config etc/kafka/client-ssl.properties 
```

# SASL/Plain authentication
[SASL/Plain authentication](https://docs.confluent.io/current/kafka/authentication_sasl/authentication_sasl_plain.html)
enables authentication based on a username and password based mechanism. SASL/Plain
authentication is recommended to be used with SSL based encryption so that the
username/password credentials and subsequent data transfer is confidential. This section
provides instructions for setting up SASL/Plain authentication with SSL encryption for Kafka
communication, as supported in DeepStream.

Make sure to first follow instructions in the **Broker Setup** section before proceeding with
these steps. 

## Configure broker settings to use SASL/Plain
Configure Kafka broker to use SASL/Plain based on instructions in the
[Configure Brokers](https://docs.confluent.io/current/security/security_tutorial.html#configure-brokers)
section of the [Confluent Quickstart security tutorial](https://docs.confluent.io/current/security/security_tutorial.html)).

The main configuration file for Kafka is *server.properties* located within *etc/kafka*.
Refer to instructions for "encrypt via SSL and authenticate via SASL". The relevant modified
contents of server.properties is shown below. User should modify fields such as paths and
password as appropriate while using this as reference.
```
listeners=SSL://:9093,SASL_SSL://:9094
security.protocol=SASL_SSL
security.inter.broker.protocol=SSL
ssl.client.auth=required
ssl.truststore.location=/var/ssl/private/kafka.server1.truststore.jks
ssl.truststore.password=test1234
ssl.keystore.location=/var/ssl/private/kafka.server1.keystore.jks
ssl.keystore.password=test1234
ssl.key.password=test1234

sasl.enabled.mechanisms=PLAIN

authorizer.class.name=kafka.security.auth.SimpleAclAuthorizer
super.users=User:kafkabroker
allow.everyone.if.no.acl.found=true
```
Note:
- change username and password appropriately.
- secure communication between Kafka brokers is configured to use SSL: user can choose to
disable this. 
- For sake of simplicity, the example enables authorization for authenticated users to access
Kafka broker topics if no relevant ACL is found based on the allow.everyone.if.no.acl.found
entry. The user should modify this to define ACL rules to suit their needs before deploying their
broker. Refer to [Authorization for Kafka broker](https://docs.confluent.io/current/kafka/authorization.html)
 documentation for details.

### Create JAAS config file
Kafka uses the JAAS (Java Authentication and Authorization Service) for SASL configuration.
JAAS file specifies the username and password credentials for authentication. Refer
[here](https://docs.confluent.io/current/kafka/authentication_sasl/index.html) for more
information.  
A sample JAAS file is provided below:
```
KafkaServer {
   org.apache.kafka.common.security.plain.PlainLoginModule required
   username="kafkabroker"
   password="kafkabroker-secret"
   user_kafkabroker="kafkabroker-secret"
   user_client="client-secret1";
};
```
It defines two users - *kafkabroker* and *client* along with their passwords.  
Define the **KAFKA_OPTS** environment variable to use the JAAS file created above
```
export KAFKA_OPTS="-Djava.security.auth.login.config=/etc/kafka/kafka_server_jaas.conf"
```
Change the path to match your situation.

## Copy the broker CA certificate to the client node
Copy the broker CA certificate created in the *Broker Setup* section (ca-cert) to the client
node and note the path where it is stored.

## DeepStream Application Changes and Configuration
The Kafka test application needs to be modified to use as part of the connection operation
the correct broker address and port used by the broker for SASL/Plain  as configured in
server.properties file.
```
conn_handle = msgapi_connect_ptr((char *)"yourserver.yourdomain.net;9094",(nvds_msgapi_connect_cb_t) sample_msgapi_connect_cb, (char *)CFG_FILE);
 ```

Kafka support in DeepStream is built around the librdkafka library. The Kafka configuration
options provided by the Kafka message adaptor to the librdkafka library needs to be modified
for SASL/Plain authentication. The DeepStream documentation describes various mechanisms to
provide these config options, but this document addresses these steps based on using a
dedicated config file name config.txt. Note that the config file used by the Kafka test
application is defined at the top of the file as part of the CFG_FILE macro.

A few parameters need to be defined using the proto-cfg entry within the message-broker
section in this config file as shown in the example below.
```
[message-broker]
proto-cfg = "security.protocol=sasl_ssl;sasl.mechanism=PLAIN;sasl.username=<username>;sasl.password=<password>;ssl.ca.location=<path to broker ca>/ca-cert;debug=broker,security"
```
The various options specified in the config file are described below:  
```security.protocol=sasl_ssl```: use SASL with SSL authentication    
```sasl.mechanism=PLAIN```: use SASL Plain mechanism for authentication  
```ssl.ca.location```: path where you copied broker certificate  
```sasl.username```: username as configured in server.properties on the broker  
```sasl.password```: password as configured in server.properties on the broker  
```debug```: enable debug logs for selected categories  


### Run the application
Run the application as before:
```
./test_kafka_proto_sync
```

If all went well, you should see logs relating to connection being established, SSL
handshake, SASL Plain authentication and ultimately the messages being sent.

## Viewing messages
You can run a Kafka consumer to receive messages being sent. 
Confluent Kafka is distributed with a Kafka consumer named kafka-console-consumer that you
can use for testing based on the command line below:
```
bin/kafka-console-consumer --bootstrap-server <address>:9094 --topic <your topic> --consumer.config etc/kafka/client-ssl.properties 
```
Again, make sure to use the correct broker port for SASL (9094)



# References
Kafka security tutorial:[https://docs.confluent.io/current/security/security_tutorial.html](https://docs.confluent.io/current/security/security_tutorial.html)

Authorization using ACLs:[https://docs.confluent.io/current/kafka/authorization.html](https://docs.confluent.io/current/kafka/authorization.html)

Rdkafka Configuration:[https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md](https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md)

Using SASL with librdkafka:[https://github.com/edenhill/librdkafka/wiki/Using-SASL-with-librdkafka](https://github.com/edenhill/librdkafka/wiki/Using-SASL-with-librdkafka)
