import fs from 'node:fs';

class TrieNode {
  constructor() {
    this.children = {};
    this.isEndOfWord = false;
    this.value = null;
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
  }

  insert(word, value) {
    let node = this.root;
    for (const char of word) {
      if (!node.children[char]) {
        node.children[char] = new TrieNode();
      }
      node = node.children[char];
    }
    node.isEndOfWord = true;
    node.value = value;
  }

  longestPrefixMatch(word) {
    let node = this.root;
    let lastMatch = null;
    let currentPrefix = '';

    for (const char of word) {
      if (!node.children[char]) break;
      currentPrefix += char;
      node = node.children[char];
      if (node.isEndOfWord) {
        lastMatch = currentPrefix;
      }
    }

    return lastMatch;
  }
}

class BPE {
  constructor(model_name, verbose = false) {
    this.model_name = model_name;
    this.verbose = verbose;
    this.trained = false;
    this.model = this._load_vocabulary();
    this.trie = new Trie();
    for (const [key, value] of Object.entries(this.model)) {
      this.trie.insert(key, value);
    }
  }

  _load_vocabulary() {
    if (fs.existsSync(this.model_name)) {
      this.trained = true;
      return JSON.parse(fs.readFileSync(this.model_name, { encoding: 'utf8' }));
    } else {
      this.trained = false;
      return {};
    }
  }

  train(corpus, save = false, k = 500) {
    if (typeof corpus !== 'string') {
      throw new Error('Cannot train nonstring');
    }

    let tokens = corpus.replace(/%/g, '').split('');
    const pairFreqs = {};
    const tokenPairs = [];

    for (let i = 0; i < tokens.length - 1; i++) {
      const pair = tokens[i] + tokens[i + 1];
      pairFreqs[pair] = (pairFreqs[pair] || 0) + 1;
      tokenPairs.push([i, i + 1]);
    }

    const heap = new MaxHeap();
    for (const [pair, freq] of Object.entries(pairFreqs)) {
      heap.insert([-freq, pair]);
    }

    for (let i = 0; i < k; i++) {
      if (heap.isEmpty()) break;

      const [negFreq, best] = heap.extractMax();
      const freq = -negFreq;
      if (freq === 1) break;

      const newToken = best.replace('%', '');
      this.model[newToken] = Object.keys(this.model).length + 1;
      this.trie.insert(newToken, this.model[newToken]);

      const newTokenPairs = [];
      for (let j = 0; j < tokenPairs.length; j++) {
        const [left, right] = tokenPairs[j];
        if (tokens[left] + tokens[right] === newToken) {
          tokens[left] = newToken;
          tokens.splice(right, 1);

          if (j > 0) {
            const prevPair = tokens[tokenPairs[j - 1][0]] + tokens[left];
            pairFreqs[prevPair] = (pairFreqs[prevPair] || 0) + 1;
            heap.insert([-pairFreqs[prevPair], prevPair]);
            newTokenPairs.push([tokenPairs[j - 1][0], left]);
          }
          if (j < tokenPairs.length - 1) {
            const nextPair = tokens[left] + tokens[tokenPairs[j + 1][1]];
            pairFreqs[nextPair] = (pairFreqs[nextPair] || 0) + 1;
            heap.insert([-pairFreqs[nextPair], nextPair]);
            newTokenPairs.push([left, tokenPairs[j + 1][1]]);
          }
        } else {
          newTokenPairs.push([left, right]);
        }
      }
      tokenPairs.length = 0;
      tokenPairs.push(...newTokenPairs);
    }

    if (save) {
      fs.writeFileSync(this.model_name, JSON.stringify(this.model));
    }

    return this.model;
  }

  tokenize(data) {
    const tokens = [];
    let i = 0;
    while (i < data.length) {
      const longestMatch = this.trie.longestPrefixMatch(data.slice(i));
      if (longestMatch) {
        tokens.push(longestMatch);
        i += longestMatch.length;
      } else {
        tokens.push(data[i]);
        i++;
      }
    }
    return tokens;
  }
}

class MaxHeap {
  constructor() {
    this.heap = [];
  }

  insert(item) {
    this.heap.push(item);
    this.bubbleUp(this.heap.length - 1);
  }

  extractMax() {
    if (this.isEmpty()) throw new Error('Heap is empty');
    const max = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.bubbleDown(0);
    }
    return max;
  }

  isEmpty() {
    return this.heap.length === 0;
  }

  bubbleUp(index) {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      if (this.heap[parentIndex][0] >= this.heap[index][0]) break;
      [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
      index = parentIndex;
    }
  }

  bubbleDown(index) {
    while (true) {
      let largest = index;
      const leftChild = 2 * index + 1;
      const rightChild = 2 * index + 2;

      if (leftChild < this.heap.length && this.heap[leftChild][0] > this.heap[largest][0]) {
        largest = leftChild;
      }
      if (rightChild < this.heap.length && this.heap[rightChild][0] > this.heap[largest][0]) {
        largest = rightChild;
      }

      if (largest === index) break;

      [this.heap[index], this.heap[largest]] = [this.heap[largest], this.heap[index]];
      index = largest;
    }
  }
}

const bpe = new BPE('abc.json');

const start = performance.now();
bpe.train(fs.readFileSync('out.txt', { encoding: 'utf8' }), true);
const end = performance.now();
const seconds = (end - start) / 1000;
console.log(`BPE: ${seconds.toFixed(3)} seconds`);
