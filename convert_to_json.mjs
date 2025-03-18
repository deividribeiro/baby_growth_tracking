/* import Realm from 'realm';

const OlamDBSchema = {
  name: 'OLAM_DB',
  primaryKey: '_id',
  properties: {
    _id: 'string',
    english_word: 'string',
    part_of_speech: 'string?',
    malayalam_definition: 'string',
  },
};

const realm = await Realm.open({
  path: 'testdata',
  schema: [OlamDBSchema],
});

let dict = realm.objects('OLAM_DB');

console.log('realm', realm);
console.log('dict', dict);
*/

import Realm from 'realm';
//const Realm = require('realm');
import fs from 'fs';
//const fs = require('fs');

// Get command-line arguments
const [, , inputFile, outputFile] = process.argv;

if (!inputFile || !outputFile) {
  console.error('Usage: node program.mjs <input_realm_file> <output_json_file>');
  process.exit(1);
}

Realm.open(inputFile)
  .then(realm => {
    const allSchemas = realm.schema;
    const exportData = {};
    
//     allSchemas.forEach(schema => {
//       const objectName = schema.name;
//       const objects = realm.objects(objectName);
//       exportData[objectName] = JSON.parse(JSON.stringify(objects));
//       // console.log('objectName',objectName);
//       // console.log('objects',objects);

//     });
    console.log('Available schemas:', allSchemas.map(schema => ({
      name: schema.name,
      properties: Object.keys(schema.properties)
    })));

    allSchemas.forEach(schema => {
      const objectName = schema.name;
      const objects = realm.objects(objectName);
      exportData[objectName] = objects.map(obj => {
        const plainObject = {};
        Object.keys(schema.properties).forEach(prop => {
          if (typeof obj[prop] === 'object' && obj[prop] !== null) {
            if (obj[prop].constructor.name === 'List') {
              plainObject[prop] = Array.from(obj[prop]);
            } else {
              plainObject[prop] = JSON.parse(JSON.stringify(obj[prop]));
            }
          } else {
            plainObject[prop] = obj[prop];
          }
        });
        return plainObject;
      });
    });

    fs.writeFileSync(outputFile, JSON.stringify(exportData, null, 2));
    console.log(`Database exported successfully to ${outputFile}`);
    realm.close();
    process.exit(0);
  })
  .catch(error => {
    console.error('Error exporting database:', error);
    process.exit(1);
  });
