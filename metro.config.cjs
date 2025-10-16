// Metro configuration for Expo
// Adds support for requiring .pte and .bin files as assets
// e.g., const model = require('./models/nllb_distilled_600M.pte');

// Use CommonJS for Metro config compatibility
const { getDefaultConfig } = require('expo/metro-config');
const exclusionList = require('metro-config/src/defaults/exclusionList');

const projectRoot = process.cwd();
const config = getDefaultConfig(projectRoot);

// Ensure assetExts contains 'pte' and 'bin' without duplicates
const extraAssetExts = ['pte', 'bin'];
config.resolver.assetExts = Array.from(
  new Set([...(config.resolver.assetExts || []), ...extraAssetExts])
);

// Exclude very large local model files from Metro's graph
config.resolver.blockList = exclusionList([
  /nllb_distilled_600M\.pte/,
  /m2m100_418M_int8\.pte/,
]);

module.exports = config;
