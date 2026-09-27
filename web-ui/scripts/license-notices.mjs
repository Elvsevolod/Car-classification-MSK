// Preserve dependency license texts in the built offline frontend.
import { readFileSync, readdirSync, existsSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const lock = JSON.parse(readFileSync('package-lock.json', 'utf8'))
const sections = ['Third-party notices for the exact installed frontend build.\n']
for (const [directory, entry] of Object.entries(lock.packages)) {
  if (!directory || !existsSync(join(directory, 'package.json'))) continue
  const pkg = JSON.parse(readFileSync(join(directory, 'package.json'), 'utf8'))
  sections.push(`\n=== ${pkg.name}@${entry.version} ===\nLicense: ${pkg.license ?? 'See included license text'}\n`)
  for (const name of readdirSync(directory, { withFileTypes: true })) {
    if (name.isFile() && /^(licen[cs]e|notice|copyright|ofl)(\.|$)/i.test(name.name)) {
      sections.push(readFileSync(join(directory, name.name), 'utf8'))
    }
  }
}
writeFileSync('dist/THIRD_PARTY_NOTICES.txt', sections.join('\n'))
