// jest.config.js
module.exports = {
    testEnvironment: "jsdom",
    verbose: true,
    roots: ["<rootDir>/lukhas/identity/frontend/pages"],
    moduleFileExtensions: ["js", "jsx"],
    transform: {
        "^.+\\.[jt]sx?$": "babel-jest"
    },
};
